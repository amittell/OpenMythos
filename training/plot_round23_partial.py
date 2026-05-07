#!/usr/bin/env python3
"""
Reconstruct round-2.3 training curve and per-T loss PNGs from the
conversation-transcript samples that survived /tmp/train_r0.log truncation.

Coverage is partial (steps 5-1470 of 3051; the second half of training
was overwritten when round 2.4 launched). The captioning notes the
limitation; final-step values are pinned from the eval JSONs.

Outputs:
    docs/paper/figures/round23_training_curve.png
    docs/paper/figures/round23_loss_by_T.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRANSCRIPT = Path(
    "/Users/alex/.claude/projects/-Users-alex-git-OpenMythos/"
    "fa7b9768-017c-4776-9282-7c360d14a564.jsonl"
)
DEPTH_EVAL = Path("/Users/alex/git/OpenMythos/docs/depth_extrap_round23.json")
OUT_DIR = Path("/Users/alex/git/OpenMythos/docs/paper/figures")
TOTAL_STEPS = 3051

STEP_RE = re.compile(
    r"step\s+(\d+)/3051\s*\|\s*loss\s+([\d.]+)\s*\|\s*ce\s+([\d.]+)"
    r"\s*\|\s*kl\s+([\d.]+)\s*\|\s*p_1\s+([\d.]+)\s*\|\s*gnorm\s+([\d.]+)"
    r"\s*\|\s*lr\s+([\d.e+-]+)\s*\|\s*T\s+(\d+)"
)


def load_samples() -> dict[str, np.ndarray]:
    """Parse the conversation transcript for unique step records."""
    by_step: dict[int, tuple] = {}
    with open(TRANSCRIPT, errors="replace") as f:
        for line in f:
            for m in STEP_RE.finditer(line):
                step = int(m.group(1))
                if step in by_step:
                    continue
                by_step[step] = (
                    float(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                    float(m.group(5)),
                    float(m.group(6)),
                    float(m.group(7)),
                    int(m.group(8)),
                )
    keys = sorted(by_step)
    return {
        "step": np.array(keys),
        "loss": np.array([by_step[k][0] for k in keys]),
        "ce": np.array([by_step[k][1] for k in keys]),
        "kl": np.array([by_step[k][2] for k in keys]),
        "p_1": np.array([by_step[k][3] for k in keys]),
        "gnorm": np.array([by_step[k][4] for k in keys]),
        "lr": np.array([by_step[k][5] for k in keys]),
        "T": np.array([by_step[k][6] for k in keys]),
    }


def final_eval_ce() -> float | None:
    """Pin the right-edge anchor from depth-extrap K=4 ACT-off CE."""
    if not DEPTH_EVAL.exists():
        return None
    with open(DEPTH_EVAL) as f:
        d = json.load(f)
    try:
        return float(d["fineweb_edu"]["act_off"]["4"]["ce"])
    except (KeyError, TypeError):
        for key in ("ce_4", "K4", "k_4", 4, "4"):
            try:
                v = d
                for part in ("fineweb_edu", "act_off", key, "ce"):
                    v = v[part] if part in v else v
                return float(v)
            except Exception:
                continue
    return None


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Right-aligned rolling mean; pads NaN at the head."""
    if len(x) < window:
        return np.full_like(x, np.nan, dtype=float)
    csum = np.cumsum(np.insert(x, 0, 0))
    out = (csum[window:] - csum[:-window]) / window
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, out])


def plot_training_curve(d: dict, out: Path, anchor_ce: float | None) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    sc = ax.scatter(
        d["step"], d["ce"], c=d["T"], cmap="viridis",
        s=18, alpha=0.7, label="ce per step (color = T)"
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("T (recurrent iterations this step)")

    ax.plot(d["step"], d["loss"], color="grey", alpha=0.3,
            linewidth=0.7, label="total loss (= ce + lambda_kl * KL)")

    window = max(8, len(d["ce"]) // 10)
    smooth = rolling_mean(d["ce"], window)
    ax.plot(d["step"], smooth, color="red", linewidth=1.8,
            label=f"{window}-pt rolling mean (ce)")

    if anchor_ce is not None:
        ax.scatter([TOTAL_STEPS], [anchor_ce], color="black", marker="X",
                   s=80, zorder=5,
                   label=f"step {TOTAL_STEPS} eval CE (K=4 ACT-off) = {anchor_ce:.3f}")

    ax.axvspan(d["step"][-1], TOTAL_STEPS, alpha=0.08, color="red")
    ax.text(
        (d["step"][-1] + TOTAL_STEPS) / 2,
        ax.get_ylim()[1] * 0.95,
        "trajectory not recovered\n(log truncated by round 2.4 launch)",
        ha="center", va="top", fontsize=9, color="darkred", alpha=0.85,
    )

    ax.set_xlim(0, TOTAL_STEPS)
    ax.set_xlabel("step")
    ax.set_ylabel("CE / loss (nats)")
    ax.set_title(
        "Round 2.3 training trajectory (joint PonderNet KL, lambda_p=0.2)\n"
        f"reconstructed from {len(d['step'])} transcript samples; "
        f"covers steps {d['step'][0]}-{d['step'][-1]} of {TOTAL_STEPS}"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def plot_loss_by_T(d: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    Ts = sorted(set(int(t) for t in d["T"]))
    means, q25, q75, ns = [], [], [], []
    for t in Ts:
        sel = d["T"] == t
        ce_t = d["ce"][sel]
        means.append(ce_t.mean())
        q25.append(np.percentile(ce_t, 25))
        q75.append(np.percentile(ce_t, 75))
        ns.append(int(sel.sum()))

    means = np.array(means)
    q25 = np.array(q25)
    q75 = np.array(q75)

    ax.fill_between(Ts, q25, q75, alpha=0.25, color="steelblue",
                    label="IQR (25th-75th pct)")
    ax.plot(Ts, means, marker="o", color="steelblue",
            linewidth=1.8, label="mean CE")

    for t, m, n in zip(Ts, means, ns):
        ax.annotate(f"n={n}", (t, m), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="dimgrey")

    ax.set_xlabel("T (sampled recurrent iterations per training step)")
    ax.set_ylabel("CE (nats)")
    ax.set_title(
        "Round 2.3 mean CE by sampled T (across recovered samples)\n"
        f"steps {d['step'][0]}-{d['step'][-1]}; "
        "flatness across T = backbone uses depth uniformly"
    )
    ax.set_xticks(Ts)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = load_samples()
    if d["step"].size == 0:
        raise SystemExit("no step samples recovered from transcript")
    anchor = final_eval_ce()
    print(
        f"recovered {d['step'].size} samples spanning step "
        f"{d['step'][0]} to {d['step'][-1]} (total run = {TOTAL_STEPS})"
    )
    if anchor is not None:
        print(f"final-step anchor CE (depth_extrap K=4 ACT-off) = {anchor:.4f}")
    plot_training_curve(d, OUT_DIR / "round23_training_curve.png", anchor)
    plot_loss_by_T(d, OUT_DIR / "round23_loss_by_T.png")


if __name__ == "__main__":
    main()
