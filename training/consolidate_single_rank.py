#!/usr/bin/env python3
"""
consolidate_single_rank.py

Convert a single-rank FSDP sharded checkpoint into a plain state dict
that downstream eval scripts can load with torch.load() without needing
to initialize a process group.

Background: when 3b_varT_act_v3.py runs single-GPU on a Blackwell box,
FSDP wraps the model with NO_SHARD strategy. The save_checkpoint()
emits ShardedTensor objects that fail to load via torch.load() unless
the loader has an active process group. Eval scripts run as standalone
single-process programs and don't have one, so they bail with:
    RuntimeError: Need to initialize default process group using
                  "init_process_group" before loading ShardedTensor

This script bridges the gap: launches a 1-rank gloo process group,
loads the ShardedTensor-bearing checkpoint, calls .local_tensor() on
each sharded entry to extract the materialized weights, then writes a
plain { name -> Tensor } dict to a new file. The output loads cleanly
with torch.load() in any context.

Usage:
    python3 consolidate_single_rank.py <input.pt> <output.pt>
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


_STATS = {"sharded": 0, "plain_tensor": 0, "scalar": 0, "container": 0}


def _unwrap_sharded(v):
    """Convert a ShardedTensor (or container holding them) into plain tensors.
    Recursive: walks dicts, lists, tuples. Leaves non-sharded values alone.

    NB: ShardedTensor extends torch.Tensor in recent PyTorch, so torch.is_tensor()
    returns True for it. We MUST check for the local_tensor/local_shards methods
    BEFORE the is_tensor branch, otherwise sharded entries pass through unchanged.
    """
    # ShardedTensor / DTensor: must be checked first since they subclass Tensor
    if hasattr(v, "local_tensor") and callable(getattr(v, "local_tensor")):
        try:
            t = v.local_tensor()
            _STATS["sharded"] += 1
            # Return a plain torch.Tensor, not a wrapper
            return torch.as_tensor(t.detach().cpu().contiguous())
        except Exception:
            pass
    if hasattr(v, "local_shards") and callable(getattr(v, "local_shards")):
        try:
            shards = v.local_shards()
            if len(shards) == 1:
                _STATS["sharded"] += 1
                return torch.as_tensor(shards[0].tensor.detach().cpu().contiguous())
            if len(shards) == 0:
                _STATS["sharded"] += 1
                meta = getattr(v, "metadata", lambda: None)()
                if meta is not None:
                    try:
                        return torch.zeros(meta.size, dtype=meta.tensor_properties.dtype)
                    except Exception:
                        pass
                return torch.empty(0)
            _STATS["sharded"] += 1
            return torch.as_tensor(torch.cat([s.tensor for s in shards], dim=0).detach().cpu().contiguous())
        except Exception:
            pass
    # Plain torch.Tensor (after sharded-detection): pass through
    if torch.is_tensor(v):
        # Detect sharded subclass that survived the above (DTensor variants etc.)
        cls_name = type(v).__name__
        if "Shard" in cls_name or "DTensor" in cls_name:
            # Re-cast to a plain tensor by copying through CPU
            try:
                _STATS["sharded"] += 1
                return torch.as_tensor(v.detach().cpu().contiguous())
            except Exception:
                pass
        _STATS["plain_tensor"] += 1
        return v
    if isinstance(v, dict):
        _STATS["container"] += 1
        return {k: _unwrap_sharded(val) for k, val in v.items()}
    if isinstance(v, list):
        _STATS["container"] += 1
        return [_unwrap_sharded(x) for x in v]
    if isinstance(v, tuple):
        _STATS["container"] += 1
        return tuple(_unwrap_sharded(x) for x in v)
    _STATS["scalar"] += 1
    return v


def consolidate(src: str, dst: str) -> None:
    if not os.path.exists(src):
        print(f"error: input does not exist: {src}", file=sys.stderr)
        sys.exit(1)

    # ShardedTensor objects need an active process group for deserialization,
    # so spin up a single-rank gloo group before torch.load.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29597")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    print(f"loading {src}")
    state = torch.load(src, map_location="cpu", weights_only=False)

    print("unwrapping ShardedTensors recursively")
    out = _unwrap_sharded(state)
    print(f"  plain tensors:    {_STATS['plain_tensor']}")
    print(f"  sharded tensors:  {_STATS['sharded']}")
    print(f"  containers:       {_STATS['container']}")
    print(f"  scalar/other:     {_STATS['scalar']}")

    tmp = dst + ".tmp"
    print(f"saving {dst}")
    torch.save(out, tmp)
    os.replace(tmp, dst)
    print(f"done: {dst} ({os.path.getsize(dst) / 1e9:.2f} GB)")

    dist.destroy_process_group()


def main():
    if len(sys.argv) != 3:
        print("usage: consolidate_single_rank.py <input.pt> <output.pt>", file=sys.stderr)
        sys.exit(2)
    consolidate(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
