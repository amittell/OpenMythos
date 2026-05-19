#!/usr/bin/env python3
"""
Streaming FSDP shard consolidator: 4 rank shards -> one full-state-dict
checkpoint, in a single non-distributed process, on commodity RAM.

WHY
---
``consolidate_ckpt.py`` (cluster, 4 procs across 4 nodes) and
``consolidate_ckpt_single_host.py`` (4 procs colocated, gloo, single host)
both wrap a model in FSDP and call ``FSDP.state_dict_type(...,
FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True))``
to gather. That gather has a ~295 GB virtual-memory peak on rank 0 -- it
materialises the full state dict plus staging buffers in rank-0's pinned
CPU memory before applying the rank0_only filter. We hit Linux's global
OOM killer at 124 GB physical (kebab-rtx6000) on 2026-05-17.

This script avoids FSDP entirely. The saved shard format (FSDP's
``SHARDED_STATE_DICT``) is already a plain ``dict[str, ShardedTensor]``
with per-parameter ``local_shards()[i].metadata.shard_offsets/shard_sizes``
encoding the rank's slice. We can assemble the full tensor by streaming
one rank file at a time, copying each shard's slice into a pre-allocated
target tensor, then freeing the rank's data before loading the next.

Peak RAM: full target dict (~6 GB bf16 for the 3.1 B model) + ONE rank
shard (~11 GB) = ~17 GB. Fits anywhere.

The output is byte-for-byte equivalent to what ``consolidate_ckpt.py``
emits (verified against the 4 eval consumers: per_token_halt_analysis.py,
depth_extrap.py, reasoning_eval.py, gen_samples_multidepth.py -- all read
``ckpt["step"]``, ``ckpt["model"]``, ``ckpt["cfg"]`` (the dataclass, not a
dict), ``ckpt["vocab_size"]``; nothing else).

Trust model: the shard files are trusted internal artifacts produced by
our own training script. ``torch.load(weights_only=False)`` is mandatory
because ``cfg`` is a pickled dataclass instance the eval scripts read by
attribute access (``getattr(cfg, "max_loop_iters", ...)``).

ShardedTensor.__setstate__ monkey-patch
---------------------------------------
``torch.load`` of a saved ``ShardedTensor`` rebuilds it through
``__setstate__``, which validates that the current process group's
world-size matches the world-size at save time. A non-distributed loader
has no PG, so the call raises ``RuntimeError: Need to initialize default
process group``. We patch ``__setstate__`` to skip the PG check and set
the unpacked tuple onto the instance directly -- the only fields we read
afterwards are ``_local_shards`` and ``_metadata`` (via ``local_shards()``
and ``metadata()``), neither of which need a live PG.

Usage:
    python3 training/consolidate_ckpt_streaming.py SRC_PATTERN DST_PATH

    SRC_PATTERN  Path prefix without ``_rankR.pt`` suffix; all rank shards
                 (0..N-1) must be present at ``{SRC_PATTERN}_rank{R}.pt``
    DST_PATH     Where the consolidated full state dict is written

Example:
    python3 training/consolidate_ckpt_streaming.py \\
        checkpoints_3b_varT_pondernet_round215/step_0007400 \\
        checkpoints_3b_varT_pondernet_round215/step_0007400_full.pt
"""

from __future__ import annotations

import glob
import os
import re
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from torch.distributed._shard.sharded_tensor import ShardedTensor


def _install_sharded_tensor_unpickle_patch() -> None:
    """Allow ``torch.load`` to deserialize ShardedTensor outside a process group.

    The default ``__setstate__`` validates that ``dist.get_world_size()``
    matches the world size at save time. We bypass that check and assign
    the unpacked state directly; we only access ``local_shards()`` and
    ``metadata()`` afterwards, which don't need a live PG.
    """

    def _safe_setstate(self, state):  # noqa: ANN001 -- mirrors torch's signature
        # state layout from torch.distributed._shard.sharded_tensor.api in
        # torch 2.0+: a 5-tuple returned by __reduce_ex__:
        #   (local_shards, sharded_tensor_metadata, process_group_state,
        #    sharding_spec, init_rrefs)
        local_shards, sharded_tensor_metadata, _pg_state, sharding_spec, init_rrefs = state
        self._local_shards = local_shards
        self._metadata = sharded_tensor_metadata
        self._sharding_spec = sharding_spec
        self._init_rrefs = init_rrefs
        # _process_group intentionally left unset -- we never invoke collectives.

    ShardedTensor.__setstate__ = _safe_setstate


def _discover_rank_files(src_pattern: str) -> list[str]:
    """Find ``{src_pattern}_rank{N}.pt`` files; return in numeric rank order.

    Lexicographic sort breaks for ranks >= 10 (``rank10`` between ``rank1``
    and ``rank2``); sort by integer suffix.
    """
    pattern = f"{src_pattern}_rank*.pt"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"no rank shards found matching {pattern}")
    rank_re = re.compile(r"_rank(\d+)\.pt$")

    def rank_key(p: str) -> int:
        m = rank_re.search(p)
        if not m:
            raise ValueError(f"path does not match rank pattern: {p}")
        return int(m.group(1))

    return sorted(files, key=rank_key)


def _shard_slice(shard) -> tuple[slice, ...]:  # noqa: ANN001 -- torch.Shard, type not exported cleanly
    """Build slices addressing the shard's region in the global tensor."""
    offsets = shard.metadata.shard_offsets
    sizes = shard.metadata.shard_sizes
    return tuple(slice(o, o + s) for o, s in zip(offsets, sizes, strict=True))


def consolidate_streaming(rank_files: list[str], dst_path: str) -> None:
    """Stream-merge FSDP rank shards into a single full-state-dict checkpoint.

    Memory model:
      - Pre-allocate ``full[k] = torch.empty(global_shape, global_dtype)``
        for every parameter, plus deep-copy each raw (un-sharded) tensor.
      - For each rank file: load -> iterate -> copy slices -> ``del`` -> GC.
      - Peak resident = sizeof(full state dict) + sizeof(one rank shard).
    """
    _install_sharded_tensor_unpickle_patch()

    t_start = time.monotonic()
    n_ranks = len(rank_files)
    logger.info(f"streaming consolidate: {n_ranks} rank shards -> {dst_path}")

    # ---- Pass 1: load rank 0, allocate target dict + capture sidecar metadata ----
    logger.info(f"[pass 1/2] loading rank 0: {rank_files[0]}")
    t0 = time.monotonic()
    ckpt0 = torch.load(rank_files[0], map_location="cpu", weights_only=False)
    logger.info(f"  loaded in {time.monotonic() - t0:.1f}s")

    sharded_state = ckpt0["model"]
    n_sharded = sum(1 for v in sharded_state.values() if isinstance(v, ShardedTensor))
    n_raw = len(sharded_state) - n_sharded
    logger.info(f"  {len(sharded_state)} entries: {n_sharded} ShardedTensor, {n_raw} raw Tensor")

    full: dict[str, torch.Tensor] = {}
    bytes_target = 0
    for k, v in sharded_state.items():
        if isinstance(v, ShardedTensor):
            meta = v.metadata()
            global_shape = tuple(meta.size)
            dtype = meta.tensor_properties.dtype
            full[k] = torch.empty(global_shape, dtype=dtype)
            bytes_target += full[k].element_size() * full[k].numel()
            # Rank 0's local slice gets copied in this same pass to avoid
            # re-loading rank 0 in pass 2.
            for shard in v.local_shards():
                full[k][_shard_slice(shard)].copy_(shard.tensor)
        else:
            # Raw tensor (rope freqs_cis caches etc.) -- identical across all
            # ranks (FSDP doesn't shard non-parameter buffers). Clone once.
            full[k] = v.detach().clone()
            bytes_target += full[k].element_size() * full[k].numel()

    step = ckpt0.get("step", 0)
    cfg = ckpt0["cfg"]  # dataclass instance; preserve as-is for eval scripts
    vocab_size = int(ckpt0.get("vocab_size", getattr(cfg, "vocab_size", 0)))
    saved_world = ckpt0.get("world_size", n_ranks)
    if saved_world != n_ranks:
        logger.warning(
            f"shard world_size={saved_world} but found {n_ranks} files; "
            "missing ranks WILL produce a wrong checkpoint"
        )

    del ckpt0
    del sharded_state
    logger.info(
        f"  target dict allocated: {len(full)} entries, "
        f"{bytes_target / 1e9:.2f} GB, step={step}, vocab_size={vocab_size}"
    )

    # ---- Pass 2: stream ranks 1..N-1 into the target dict ----
    for r, rank_file in enumerate(rank_files[1:], start=1):
        logger.info(f"[pass 2/2] loading rank {r}: {rank_file}")
        t0 = time.monotonic()
        ckpt = torch.load(rank_file, map_location="cpu", weights_only=False)
        logger.info(f"  loaded in {time.monotonic() - t0:.1f}s")

        t0 = time.monotonic()
        n_copied = 0
        for k, v in ckpt["model"].items():
            if not isinstance(v, ShardedTensor):
                # Already populated from rank 0; raw buffers are identical
                # across ranks.
                continue
            for shard in v.local_shards():
                full[k][_shard_slice(shard)].copy_(shard.tensor)
                n_copied += 1
        logger.info(f"  copied {n_copied} shard slices in {time.monotonic() - t0:.1f}s")

        del ckpt

    # ---- Write the consolidated output (atomic via .tmp + os.replace) ----
    out = {
        "step": step,
        "model": full,
        "cfg": cfg,
        "vocab_size": vocab_size,
    }

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path + ".tmp"
    logger.info(f"writing to {tmp_path}")
    t0 = time.monotonic()
    torch.save(out, tmp_path)
    os.replace(tmp_path, dst_path)
    logger.success(
        f"consolidated full state dict written to {dst_path} "
        f"({time.monotonic() - t0:.1f}s write; {time.monotonic() - t_start:.1f}s total)"
    )


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "usage: consolidate_ckpt_streaming.py SRC_PATTERN DST_PATH",
            file=sys.stderr,
        )
        sys.exit(2)

    src_pattern = sys.argv[1]
    dst_path = sys.argv[2]

    rank_files = _discover_rank_files(src_pattern)
    logger.info(f"discovered {len(rank_files)} rank files: {[os.path.basename(f) for f in rank_files]}")

    consolidate_streaming(rank_files, dst_path)


if __name__ == "__main__":
    main()
