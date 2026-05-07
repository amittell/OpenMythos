#!/usr/bin/env python3
"""
Generate loss-curve plots from a training log.

Reads /tmp/train_r0.log (the rank-0 training log emitted by 3b_varT_fast.py),
parses every "step N | loss X | gnorm Y | lr Z | T K" line, and produces:

    docs/training_curve.png      loss vs step, plus a 50-step rolling mean
    docs/loss_by_T.png           per-T mean loss (with quartile range), shows
                                 whether the model fights any particular depth

Run with no args; relies on default paths.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


LOG = Path("/tmp/train_r0.log")
OUT_CURVE = Path("/home/alexm/OpenMythos/docs/training_curve.png")
OUT_BYT = Path("/home/alexm/OpenMythos/docs/loss_by_T.png")


STEP_RE = re.compile(
    r"step\s+(\d+)/\d+\s+\| loss\s+([\d.]+)\s+\| gnorm\s+([\d.]+)\s+\| lr\s+([\d.e+-]+)\s+\| T\s*(\d+)"
)


def parse(log_path: Path) -> dict:
    if not log_path.exists():
        print(f"ERROR: {log_path} not found", file=sys.stderr)
        sys.exit(1)
    steps, losses, gnorms, lrs, Ts = [], [], [], [], []
    with open(log_path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                gnorms.append(float(m.group(3)))
                lrs.append(float(m.group(4)))
                Ts.append(int(m.group(5)))
    return {
        "step": np.array(steps),
        "loss": np.array(losses),
        "gnorm": np.array(gnorms),
        "lr": np.array(lrs),
        "T": np.array(Ts),
    }


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return x
    csum = np.cumsum(np.insert(x, 0, 0))
    out = (csum[window:] - csum[:-window]) / window
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, out])


def plot_loss_curve(d: dict, out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.scatter(d["step"], d["loss"], c=d["T"], cmap="viridis", s=4, alpha=0.4)
    smooth = rolling_mean(d["loss"], 50)
    ax1.plot(d["step"], smooth, color="red", linewidth=1.5, label="50-step rolling mean")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.set_title(
        f"Round 2 training loss (steps {d['step'][0]}–{d['step'][-1]}, "
        f"min={d['loss'].min():.3f} at step {int(d['step'][np.argmin(d['loss'])])})"
    )
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=d["T"].min(), vmax=d["T"].max()),
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label="T (loop depth)")
    ax2 = ax1.twinx()
    ax2.plot(d["step"], d["lr"], color="orange", linewidth=0.8, alpha=0.5, label="lr")
    ax2.set_ylabel("learning rate", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    plt.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def plot_loss_by_T(d: dict, out: Path) -> None:
    Ts = sorted(set(d["T"].tolist()))
    means = []
    p25s = []
    p75s = []
    counts = []
    for t in Ts:
        sel = d["T"] == t
        vals = d["loss"][sel]
        means.append(vals.mean())
        p25s.append(np.percentile(vals, 25))
        p75s.append(np.percentile(vals, 75))
        counts.append(int(sel.sum()))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(Ts, means, "o-", color="C0", linewidth=2, label="mean")
    ax1.fill_between(
        Ts, p25s, p75s, alpha=0.25, color="C0", label="25-75 percentile"
    )
    ax1.set_ylabel("loss")
    ax1.set_title("Round 2: loss vs sampled T")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.bar(Ts, counts, color="C1")
    ax2.set_xlabel("T (loop depth)")
    ax2.set_ylabel("# steps sampled")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    d = parse(LOG)
    print(f"parsed {len(d['step'])} step records")
    if len(d["step"]) == 0:
        print("no step records found, skipping plots")
        return
    OUT_CURVE.parent.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(d, OUT_CURVE)
    plot_loss_by_T(d, OUT_BYT)


if __name__ == "__main__":
    main()
