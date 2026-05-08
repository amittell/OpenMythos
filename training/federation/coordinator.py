"""
Federated training coordinator.

Runs on spark.lan. Polls the federation directory for ready markers from
both cluster and RTX6000 trainers, pulls state dicts, computes a
token-weighted average, pushes the average back to both sides, and
manages crash-resumable round state.

Usage:
    python3 coordinator.py [--fed-dir DIR] [--rtx-host HOST] [--rtx-remote-dir DIR]

Environment variables (alternative to flags):
    FED_DIR            local fed working directory (default: /home/alexm/OpenMythos/fed)
    FED_RTX_HOST       SSH host for RTX6000 (default: alexm@kebab-rtx6000.lan)
    FED_RTX_REMOTE_DIR fed dir on RTX6000 (default: /home/alexm/OpenMythos/fed)
    FED_SYNC_TIMEOUT   max seconds to wait for both ready markers (default: 3600)
    FED_POLL_INTERVAL  seconds between filesystem polls (default: 10)
    FED_AVG_DTYPE      torch dtype for averaging (default: float32)
"""

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("fed.coordinator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%F %T",
)


# Phases of a single sync round
PHASE_WAITING = "WAITING"          # Waiting for both ready markers
PHASE_PULLING = "PULLING"          # Pulling state from RTX6000 (cluster is local)
PHASE_AVERAGING = "AVERAGING"      # Computing weighted average
PHASE_DISTRIBUTING = "DISTRIBUTING"  # Pushing avg to both sides
PHASE_COMPLETING = "COMPLETING"    # Waiting for both `loaded` acks
PHASE_DONE = "DONE"                # Round complete; will be reset to WAITING for next


ROLES = ("cluster", "rtx")


@dataclass
class CoordState:
    """Persisted coordinator state. Loaded on startup, saved after each phase change."""
    round: int = 0
    phase: str = PHASE_WAITING
    last_sync_unix: float = 0.0
    cluster_dead: bool = False
    rtx_dead: bool = False
    history: list = field(default_factory=list)  # list of dicts per completed round

    @classmethod
    def load_or_default(cls, path: Path) -> "CoordState":
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            return cls(**data)
        return cls()

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=2)
        os.replace(tmp, path)


def round_dir(fed_dir: Path, round_num: int) -> Path:
    return fed_dir / f"round_{round_num:04d}"


def role_state_path(round_path: Path, role: str) -> Path:
    return round_path / f"{role}_state.pt"


def role_meta_path(round_path: Path, role: str) -> Path:
    return round_path / f"{role}_meta.json"


def role_ready_path(round_path: Path, role: str) -> Path:
    return round_path / f"{role}_ready"


def role_loaded_path(round_path: Path, role: str) -> Path:
    return round_path / f"{role}_loaded"


def avg_path(round_path: Path) -> Path:
    return round_path / "avg_state.pt"


def halt_flag(fed_dir: Path) -> Path:
    return fed_dir / "halt"


def rsync_pull(remote: str, src: str, dst: str) -> bool:
    """rsync src on remote -> dst locally. Returns True on success."""
    try:
        subprocess.run(
            ["rsync", "-az", "--timeout=120", f"{remote}:{src}", dst],
            check=True,
            capture_output=True,
            timeout=600,
        )
        return True
    except subprocess.TimeoutExpired:
        logger.warning("rsync_pull timed out: %s:%s -> %s", remote, src, dst)
        return False
    except subprocess.CalledProcessError as e:
        logger.warning("rsync_pull failed (%s): %s", e.returncode, e.stderr.decode("utf-8", errors="replace")[:500])
        return False


def rsync_push(src: str, remote: str, dst: str) -> bool:
    """rsync src locally -> dst on remote. Returns True on success."""
    try:
        subprocess.run(
            ["rsync", "-az", "--timeout=120", src, f"{remote}:{dst}"],
            check=True,
            capture_output=True,
            timeout=600,
        )
        return True
    except subprocess.TimeoutExpired:
        logger.warning("rsync_push timed out: %s -> %s:%s", src, remote, dst)
        return False
    except subprocess.CalledProcessError as e:
        logger.warning("rsync_push failed (%s): %s", e.returncode, e.stderr.decode("utf-8", errors="replace")[:500])
        return False


def ssh_test_path(remote: str, path: str) -> bool:
    """Returns True if `path` exists on `remote`."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", remote, f"test -f {path} && echo YES"],
            capture_output=True,
            timeout=30,
        )
        return result.stdout.strip() == b"YES"
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


def ssh_touch(remote: str, path: str) -> bool:
    try:
        subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", remote, f"mkdir -p {os.path.dirname(path)} && touch {path}"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.warning("ssh_touch failed: %s", e)
        return False


class FederationCoordinator:

    def __init__(
        self,
        fed_dir: Path,
        rtx_host: str,
        rtx_remote_dir: Path,
        sync_timeout: int = 3600,
        poll_interval: int = 10,
        avg_dtype: torch.dtype = torch.float32,
    ):
        self.fed_dir = fed_dir
        self.rtx_host = rtx_host
        self.rtx_remote_dir = rtx_remote_dir
        self.sync_timeout = sync_timeout
        self.poll_interval = poll_interval
        self.avg_dtype = avg_dtype
        self.state_path = fed_dir / "coord.state.json"
        self.fed_dir.mkdir(parents=True, exist_ok=True)
        self.state = CoordState.load_or_default(self.state_path)
        self._stop = False
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        logger.info("Coordinator initialized; resumed at round=%d phase=%s", self.state.round, self.state.phase)

    def _handle_signal(self, signum, _frame):
        logger.info("Received signal %d; will stop after current phase", signum)
        self._stop = True

    def _save_state(self):
        self.state.save(self.state_path)

    # ------------------------------------------------------------------
    # Remote ready/loaded checks
    # ------------------------------------------------------------------

    def cluster_ready(self, round_num: int) -> bool:
        return role_ready_path(round_dir(self.fed_dir, round_num), "cluster").exists()

    def rtx_ready(self, round_num: int) -> bool:
        # RTX is on remote host; check via SSH
        remote_path = str(self.rtx_remote_dir / f"round_{round_num:04d}" / "rtx_ready")
        return ssh_test_path(self.rtx_host, remote_path)

    def cluster_loaded(self, round_num: int) -> bool:
        return role_loaded_path(round_dir(self.fed_dir, round_num), "cluster").exists()

    def rtx_loaded(self, round_num: int) -> bool:
        remote_path = str(self.rtx_remote_dir / f"round_{round_num:04d}" / "rtx_loaded")
        return ssh_test_path(self.rtx_host, remote_path)

    # ------------------------------------------------------------------
    # State pull/push
    # ------------------------------------------------------------------

    def pull_rtx_state(self, round_num: int) -> bool:
        """Pull the RTX6000 state.pt and meta.json into local round_dir."""
        rd = round_dir(self.fed_dir, round_num)
        rd.mkdir(parents=True, exist_ok=True)
        remote_state = str(self.rtx_remote_dir / f"round_{round_num:04d}" / "rtx_state.pt")
        remote_meta = str(self.rtx_remote_dir / f"round_{round_num:04d}" / "rtx_meta.json")
        ok1 = rsync_pull(self.rtx_host, remote_state, str(rd / "rtx_state.pt"))
        ok2 = rsync_pull(self.rtx_host, remote_meta, str(rd / "rtx_meta.json"))
        return ok1 and ok2

    def push_avg_to_rtx(self, round_num: int) -> bool:
        rd = round_dir(self.fed_dir, round_num)
        remote_dir = str(self.rtx_remote_dir / f"round_{round_num:04d}")
        # Ensure remote dir exists
        try:
            subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", self.rtx_host, f"mkdir -p {remote_dir}"],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.warning("Failed to mkdir on RTX6000: %s", e)
            return False
        return rsync_push(str(avg_path(rd)), self.rtx_host, f"{remote_dir}/avg_state.pt")

    # ------------------------------------------------------------------
    # Validation and averaging
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_state_dict(state: dict, name: str) -> bool:
        for k, v in state.items():
            if not torch.is_tensor(v):
                continue
            if torch.isnan(v).any():
                logger.error("[%s] tensor %s contains NaN", name, k)
                return False
            if torch.isinf(v).any():
                logger.error("[%s] tensor %s contains Inf", name, k)
                return False
        return True

    def average_states(self, round_num: int) -> Optional[dict]:
        rd = round_dir(self.fed_dir, round_num)
        cluster_state_path = rd / "cluster_state.pt"
        rtx_state_path = rd / "rtx_state.pt"
        cluster_meta_path = rd / "cluster_meta.json"
        rtx_meta_path = rd / "rtx_meta.json"

        cluster_alive = cluster_state_path.exists() and cluster_meta_path.exists()
        rtx_alive = rtx_state_path.exists() and rtx_meta_path.exists()

        if not cluster_alive and not rtx_alive:
            logger.error("Round %d: neither side has state available", round_num)
            return None

        # Load metas
        cluster_meta = json.loads(cluster_meta_path.read_text()) if cluster_alive else {"tokens_since_last_sync": 0}
        rtx_meta = json.loads(rtx_meta_path.read_text()) if rtx_alive else {"tokens_since_last_sync": 0}
        w_c = float(cluster_meta.get("tokens_since_last_sync", 0))
        w_r = float(rtx_meta.get("tokens_since_last_sync", 0))

        if cluster_alive and rtx_alive and (w_c + w_r) <= 0:
            logger.warning("Round %d: both weights zero, falling back to equal weight", round_num)
            w_c = w_r = 1.0

        logger.info(
            "Round %d weights: cluster=%.0f tokens, rtx=%.0f tokens, ratio=%.2f",
            round_num, w_c, w_r, w_c / max(1.0, w_c + w_r),
        )

        # Load states
        cluster_state = torch.load(cluster_state_path, map_location="cpu") if cluster_alive else None
        rtx_state = torch.load(rtx_state_path, map_location="cpu") if rtx_alive else None

        # Validate
        if cluster_state is not None and not self._validate_state_dict(cluster_state, "cluster"):
            logger.error("Cluster state failed validation; aborting average")
            return None
        if rtx_state is not None and not self._validate_state_dict(rtx_state, "rtx"):
            logger.error("RTX state failed validation; aborting average")
            return None

        # Solo path: only one side alive
        if cluster_state is None:
            logger.warning("Round %d: cluster missing; using RTX state as average (degraded)", round_num)
            return rtx_state
        if rtx_state is None:
            logger.warning("Round %d: rtx missing; using cluster state as average (degraded)", round_num)
            return cluster_state

        # Both alive: token-weighted average
        return self._weighted_average(cluster_state, rtx_state, w_c, w_r)

    def _weighted_average(self, sd_a: dict, sd_b: dict, w_a: float, w_b: float) -> dict:
        if set(sd_a.keys()) != set(sd_b.keys()):
            extra_a = set(sd_a.keys()) - set(sd_b.keys())
            extra_b = set(sd_b.keys()) - set(sd_a.keys())
            logger.error(
                "State dict key mismatch: only-cluster=%s only-rtx=%s",
                sorted(extra_a)[:5], sorted(extra_b)[:5],
            )
            raise ValueError("Cannot average state dicts with mismatched keys")

        total = w_a + w_b
        f_a = w_a / total
        f_b = w_b / total
        out = {}
        for k in sd_a.keys():
            t_a = sd_a[k]
            t_b = sd_b[k]
            if not torch.is_tensor(t_a) or not torch.is_tensor(t_b):
                # Non-tensor entry: copy from A (assume identical)
                out[k] = t_a
                continue
            if t_a.shape != t_b.shape:
                logger.error("Shape mismatch on %s: %s vs %s", k, t_a.shape, t_b.shape)
                raise ValueError(f"Shape mismatch on key {k}")
            if t_a.dtype != t_b.dtype:
                # Promote to common dtype, average, cast back
                target = self.avg_dtype
                avg = (t_a.to(target) * f_a + t_b.to(target) * f_b).to(t_a.dtype)
            else:
                # Average in fp32 to avoid bf16/fp16 precision loss
                if t_a.is_floating_point():
                    avg = (t_a.to(self.avg_dtype) * f_a + t_b.to(self.avg_dtype) * f_b).to(t_a.dtype)
                else:
                    # Integer tensors (rare in model state): take cluster's
                    avg = t_a
            out[k] = avg
        return out

    def write_avg(self, round_num: int, avg_state: dict) -> Path:
        rd = round_dir(self.fed_dir, round_num)
        dst = avg_path(rd)
        tmp = dst.with_suffix(".tmp")
        torch.save(avg_state, tmp)
        os.replace(tmp, dst)
        return dst

    # ------------------------------------------------------------------
    # Per-round driver
    # ------------------------------------------------------------------

    def run_round(self, round_num: int) -> bool:
        """Run a full sync round. Returns True on success, False on abort."""
        rd = round_dir(self.fed_dir, round_num)
        rd.mkdir(parents=True, exist_ok=True)
        logger.info("=== ROUND %d START (phase=%s) ===", round_num, self.state.phase)

        # Phase: WAITING for both ready markers
        if self.state.phase in (PHASE_WAITING, PHASE_DONE):
            self.state.phase = PHASE_WAITING
            self._save_state()
            deadline = time.time() + self.sync_timeout
            cluster_seen = self.cluster_ready(round_num)
            rtx_seen = self.rtx_ready(round_num)
            while not (cluster_seen and rtx_seen):
                if self._stop:
                    logger.info("Stopping requested; abandoning round %d at WAITING", round_num)
                    return False
                if halt_flag(self.fed_dir).exists():
                    logger.error("Halt flag set; aborting")
                    return False
                if time.time() > deadline:
                    # Decide who is dead
                    if not cluster_seen and not rtx_seen:
                        logger.error("Both sides missed sync window for round %d; halting", round_num)
                        return False
                    if not cluster_seen:
                        logger.warning("Cluster missed sync window; treating as dead this round")
                        self.state.cluster_dead = True
                    if not rtx_seen:
                        logger.warning("RTX missed sync window; treating as dead this round")
                        self.state.rtx_dead = True
                    self._save_state()
                    break
                time.sleep(self.poll_interval)
                cluster_seen = cluster_seen or self.cluster_ready(round_num)
                rtx_seen = rtx_seen or self.rtx_ready(round_num)
            logger.info("Round %d: ready cluster=%s rtx=%s", round_num, cluster_seen, rtx_seen)

        # Phase: PULLING (rtx state from remote)
        if self.state.phase in (PHASE_WAITING, PHASE_PULLING):
            self.state.phase = PHASE_PULLING
            self._save_state()
            if self.rtx_ready(round_num) and not self.state.rtx_dead:
                if not self.pull_rtx_state(round_num):
                    logger.error("Failed to pull RTX state for round %d", round_num)
                    # Continue with degraded mode (cluster only)
                    self.state.rtx_dead = True
                    self._save_state()

        # Phase: AVERAGING
        if self.state.phase in (PHASE_PULLING, PHASE_AVERAGING):
            self.state.phase = PHASE_AVERAGING
            self._save_state()
            if not avg_path(rd).exists():
                avg_state = self.average_states(round_num)
                if avg_state is None:
                    logger.error("Averaging failed for round %d", round_num)
                    return False
                self.write_avg(round_num, avg_state)
                logger.info("Round %d: avg written to %s", round_num, avg_path(rd))
            else:
                logger.info("Round %d: avg already exists (resumed mid-round)", round_num)

        # Phase: DISTRIBUTING (push avg to RTX; cluster reads it directly from local fed_dir)
        if self.state.phase in (PHASE_AVERAGING, PHASE_DISTRIBUTING):
            self.state.phase = PHASE_DISTRIBUTING
            self._save_state()
            if not self.state.rtx_dead:
                push_attempts = 0
                while push_attempts < 5:
                    if self.push_avg_to_rtx(round_num):
                        break
                    push_attempts += 1
                    logger.warning("push_avg_to_rtx failed (attempt %d); retrying in %ds", push_attempts, 30)
                    time.sleep(30)
                else:
                    logger.error("Failed to push avg to RTX after 5 attempts; treating as dead")
                    self.state.rtx_dead = True
                    self._save_state()

        # Phase: COMPLETING (wait for both `loaded` markers)
        if self.state.phase in (PHASE_DISTRIBUTING, PHASE_COMPLETING):
            self.state.phase = PHASE_COMPLETING
            self._save_state()
            deadline = time.time() + self.sync_timeout
            cluster_loaded = self.state.cluster_dead or self.cluster_loaded(round_num)
            rtx_loaded = self.state.rtx_dead or self.rtx_loaded(round_num)
            while not (cluster_loaded and rtx_loaded):
                if self._stop:
                    logger.info("Stopping requested; round %d not fully completed", round_num)
                    return False
                if time.time() > deadline:
                    if not cluster_loaded:
                        logger.warning("Cluster did not ack load; marking dead for next round")
                        self.state.cluster_dead = True
                    if not rtx_loaded:
                        logger.warning("RTX did not ack load; marking dead for next round")
                        self.state.rtx_dead = True
                    self._save_state()
                    break
                time.sleep(self.poll_interval)
                cluster_loaded = cluster_loaded or self.cluster_loaded(round_num)
                rtx_loaded = rtx_loaded or self.rtx_loaded(round_num)

        # Phase: DONE
        self.state.phase = PHASE_DONE
        self.state.last_sync_unix = time.time()
        self.state.history.append({
            "round": round_num,
            "completed_at": self.state.last_sync_unix,
            "cluster_dead": self.state.cluster_dead,
            "rtx_dead": self.state.rtx_dead,
        })
        # Cap history length
        if len(self.state.history) > 200:
            self.state.history = self.state.history[-200:]
        self._save_state()
        logger.info("=== ROUND %d COMPLETE ===", round_num)
        return True

    def run(self) -> None:
        logger.info("Coordinator main loop starting")
        while not self._stop:
            if halt_flag(self.fed_dir).exists():
                logger.error("Halt flag set; exiting")
                return
            success = self.run_round(self.state.round)
            if not success:
                logger.warning("Round %d incomplete; sleeping before retry", self.state.round)
                time.sleep(60)
                continue
            # Reset per-round dead flags so next round re-checks
            self.state.cluster_dead = False
            self.state.rtx_dead = False
            self.state.round += 1
            self.state.phase = PHASE_WAITING
            self._save_state()
        logger.info("Coordinator stopping (round=%d phase=%s)", self.state.round, self.state.phase)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fed-dir", default=os.environ.get("FED_DIR", "/home/alexm/OpenMythos/fed"))
    parser.add_argument("--rtx-host", default=os.environ.get("FED_RTX_HOST", "alexm@kebab-rtx6000.lan"))
    parser.add_argument("--rtx-remote-dir", default=os.environ.get("FED_RTX_REMOTE_DIR", "/home/alexm/OpenMythos/fed"))
    parser.add_argument("--sync-timeout", type=int, default=int(os.environ.get("FED_SYNC_TIMEOUT", "3600")))
    parser.add_argument("--poll-interval", type=int, default=int(os.environ.get("FED_POLL_INTERVAL", "10")))
    args = parser.parse_args()

    coord = FederationCoordinator(
        fed_dir=Path(args.fed_dir),
        rtx_host=args.rtx_host,
        rtx_remote_dir=Path(args.rtx_remote_dir),
        sync_timeout=args.sync_timeout,
        poll_interval=args.poll_interval,
    )
    coord.run()


if __name__ == "__main__":
    main()
