"""Plot Phase 2 (adaptation module) training loss + MAE curves from log file."""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ITER_RE = re.compile(r"\[iter\s+(\d+)/\d+\]\s+loss=([\d.]+)\s+mae=([\d.]+)")


def parse_log(path):
    iters, losses, maes = [], [], []
    with open(path, "r") as f:
        for line in f:
            m = ITER_RE.search(line)
            if not m:
                continue
            iters.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            maes.append(float(m.group(3)))
    return np.array(iters), np.array(losses), np.array(maes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=str, required=True)
    p.add_argument("--out", type=str, default=None,
                   help="Output PNG path (default: alongside log)")
    args = p.parse_args()

    iters, losses, maes = parse_log(args.log)
    if len(iters) == 0:
        print(f"No iter lines found in {args.log}", file=sys.stderr)
        sys.exit(1)
    print(f"Parsed {len(iters)} iter records from {args.log}")
    print(f"  iter range: {iters.min()} → {iters.max()}")
    print(f"  loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"  mae : {maes[0]:.4f} → {maes[-1]:.4f}")

    out = args.out
    if out is None:
        out = os.path.splitext(args.log)[0] + "_loss.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(iters, losses, "C0", lw=1.5, label="MSE loss")
    # smoothed (rolling mean, window=20)
    if len(losses) >= 20:
        w = min(20, len(losses) // 5)
        kernel = np.ones(w) / w
        smooth = np.convolve(losses, kernel, mode="valid")
        smooth_x = iters[w-1:]
        ax.plot(smooth_x, smooth, "C0", lw=2.5, alpha=0.5, label=f"rolling mean (w={w})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE(z_pred, z_teacher)")
    ax.set_title("Phase 2 — Adaptation module: distillation loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale("log")

    ax = axes[1]
    ax.plot(iters, maes, "C1", lw=1.5, label="MAE")
    if len(maes) >= 20:
        w = min(20, len(maes) // 5)
        kernel = np.ones(w) / w
        smooth = np.convolve(maes, kernel, mode="valid")
        ax.plot(iters[w-1:], smooth, "C1", lw=2.5, alpha=0.5, label=f"rolling mean (w={w})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MAE(z_pred, z_teacher)")
    ax.set_title("Phase 2 — per-element absolute error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("RMA Phase 2 supervised distillation training", y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
