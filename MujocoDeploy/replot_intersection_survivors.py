"""Re-plot Phase 2 tracking RMSE using intersection-of-survivors.

For each (body, magnitude) bin the original plot averaged RMSE over whichever
trials each mode survived. When baseline falls at a direction while adaptation
stays up, baseline's mean is computed only over the easy surviving directions
— so baseline can appear to have *lower* RMSE than adaptation even though it
is strictly less robust. This script restricts the mean to the set of trials
where ALL three modes (teacher, adaptation, baseline) survived, removing the
confound.

Usage:
  python replot_intersection_survivors.py \
      --csv phase2_eval_main_stress/phase2_eval.csv \
      --out phase2_eval_main_stress/phase2_tracking_intersection.png
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODES = ("teacher", "adaptation", "baseline")
MODE_COLOR = {"teacher": "C0", "adaptation": "C1", "baseline": "C3"}
MODE_MARKER = {"teacher": "o", "adaptation": "s", "baseline": "^"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--raw_out", default=None,
                        help="Also save the original survivor-only plot for side-by-side")
    args = parser.parse_args()

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.csv)),
        "phase2_tracking_intersection.png",
    )
    raw_out = args.raw_out or os.path.join(
        os.path.dirname(os.path.abspath(args.csv)),
        "phase2_tracking_survivors_only.png",
    )

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        print("empty CSV"); return

    # Index by (body, mag, direction, mode)
    by_key: dict[tuple[str, float, str, str], dict] = {}
    for r in rows:
        parts = r["label"].split("|")
        body = parts[0]
        mag = float(parts[1].replace("N", ""))
        direction = parts[2] if len(parts) > 2 else "zero"
        by_key[(body, mag, direction, r["mode"])] = r

    # Collect distinct bodies, magnitudes, directions per body
    bodies = sorted({k[0] for k in by_key.keys()})
    mag_set = sorted({k[1] for k in by_key.keys()})
    dirs_per_body_mag: dict[tuple[str, float], set[str]] = defaultdict(set)
    for (b, m, d, _) in by_key.keys():
        dirs_per_body_mag[(b, m)].add(d)

    # ---- Plot 1: intersection-of-survivors ----
    fig, axes = plt.subplots(1, len(bodies), figsize=(5 * len(bodies), 5), sharey=True)
    if len(bodies) == 1:
        axes = [axes]
    caption_lines = []
    for ax, body in zip(axes, bodies):
        mode_pts: dict[str, list[tuple[float, float, int]]] = {m: [] for m in MODES}
        for mag in mag_set:
            dirs = sorted(dirs_per_body_mag.get((body, mag), set()))
            # Intersection: direction where all 3 modes have success==True
            inter = []
            for d in dirs:
                keys = [(body, mag, d, m) for m in MODES]
                if not all(k in by_key for k in keys):
                    continue
                if all(by_key[k]["success"] == "True" for k in keys):
                    inter.append(d)
            if not inter:
                caption_lines.append(f"{body} @ {mag}N: 0 shared survivors (all modes fell or mismatched)")
                continue
            caption_lines.append(f"{body} @ {mag}N: n={len(inter)}/{len(dirs)} shared survivors")
            for mode in MODES:
                vals = [float(by_key[(body, mag, d, mode)]["tracking_rmse_xy"]) for d in inter]
                vals = [v for v in vals if not np.isnan(v)]
                if vals:
                    mode_pts[mode].append((mag, float(np.mean(vals)), len(vals)))
        for mode in MODES:
            pts = mode_pts[mode]
            if not pts:
                continue
            mags = [p[0] for p in pts]
            means = [p[1] for p in pts]
            ax.plot(mags, means, "-" + MODE_MARKER[mode], color=MODE_COLOR[mode],
                    label=mode, markersize=7)
        ax.set_xlabel("Force magnitude (N)")
        ax.set_title(body)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Tracking RMSE (m/s)")
    fig.suptitle("Phase 2 eval: tracking RMSE vs perturbation (shared survivors across all 3 modes)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # ---- Plot 2: original per-mode survivor plot for side-by-side ----
    fig, axes = plt.subplots(1, len(bodies), figsize=(5 * len(bodies), 5), sharey=True)
    if len(bodies) == 1:
        axes = [axes]
    for ax, body in zip(axes, bodies):
        for mode in MODES:
            per_mag: dict[float, list[float]] = defaultdict(list)
            for (b, m, d, mo), r in by_key.items():
                if b != body or mo != mode or r["success"] != "True":
                    continue
                v = float(r["tracking_rmse_xy"])
                if not np.isnan(v):
                    per_mag[m].append(v)
            mags = sorted(per_mag.keys())
            means = [np.mean(per_mag[m]) for m in mags]
            if mags:
                ax.plot(mags, means, "-" + MODE_MARKER[mode], color=MODE_COLOR[mode],
                        label=mode, markersize=7)
        ax.set_xlabel("Force magnitude (N)")
        ax.set_title(body)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Tracking RMSE (m/s)")
    fig.suptitle("Phase 2 eval: tracking RMSE (per-mode survivors, survivor-bias confound)")
    plt.tight_layout()
    plt.savefig(raw_out, dpi=150)
    plt.close()
    print(f"Saved: {raw_out}")

    print("\n--- Intersection diagnostics ---")
    for line in caption_lines:
        print(" ", line)


if __name__ == "__main__":
    main()
