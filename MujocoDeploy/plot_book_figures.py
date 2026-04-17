#!/usr/bin/env python3
"""Generate publication-quality figures for the RMA book chapter.

Reads the evaluation CSV and produces refined plots that separate
horizontal (challenging) and vertical (easy) perturbation directions,
plus the adaptation time-series analysis.
"""

import os
import sys
import csv
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import mujoco
from rsl_rl.modules import ActorCriticRecurrent
from rma.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(_SCRIPT_DIR, "book_eval_results")
CSV_PATH = os.path.join(OUT_DIR, "rma_eval_results.csv")

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
})

RMA_COLOR = "#1565C0"   # deep blue
BASE_COLOR = "#C62828"  # deep red
RMA_LIGHT = "#90CAF9"
BASE_LIGHT = "#EF9A9A"

HORIZONTAL_DIRS = {'+X', '-X', '+Y', '-Y', 'rnd0', 'rnd1', 'rnd2', 'rnd3', 'rnd4'}


def load_results():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def aggregate_horizontal(results, force_mags):
    """Aggregate metrics for horizontal-only forces."""
    rma_data = {"sr": [], "sr_std": [], "track": [], "track_std": [],
                "orient": [], "energy": [], "smooth": [], "surv": [], "surv_std": []}
    base_data = {"sr": [], "sr_std": [], "track": [], "track_std": [],
                 "orient": [], "energy": [], "smooth": [], "surv": [], "surv_std": []}

    for mag in force_mags:
        for method, data in [("RMA", rma_data), ("Baseline", base_data)]:
            if mag == 0:
                horiz = [r for r in results if r['method'] == method
                         and float(r['force_mag']) == mag]
            else:
                horiz = [r for r in results if r['method'] == method
                         and float(r['force_mag']) == mag
                         and r['direction'] in HORIZONTAL_DIRS]

            n = max(len(horiz), 1)
            n_success = sum(1 for r in horiz if r['success'] == 'True')
            data["sr"].append(100 * n_success / n)

            surv_times = [float(r['survival_time']) for r in horiz]
            data["surv"].append(np.mean(surv_times) if surv_times else 0)
            data["surv_std"].append(np.std(surv_times) if surv_times else 0)

            survivors = [r for r in horiz if r['success'] == 'True']
            if survivors:
                tracks = [float(r['tracking_rmse_xy']) for r in survivors
                          if r['tracking_rmse_xy'] != 'nan']
                data["track"].append(np.mean(tracks) if tracks else float('nan'))
                data["track_std"].append(np.std(tracks) if tracks else 0)
                data["orient"].append(np.mean([float(r['mean_orientation_err']) for r in survivors]))
                data["energy"].append(np.mean([float(r['mean_energy']) for r in survivors]))
                data["smooth"].append(np.mean([float(r['mean_smoothness']) for r in survivors]))
            else:
                data["track"].append(float('nan'))
                data["track_std"].append(0)
                data["orient"].append(float('nan'))
                data["energy"].append(float('nan'))
                data["smooth"].append(float('nan'))

    return rma_data, base_data


def fig1_main_comparison(results, force_mags):
    """Figure 1: Success rate + tracking error for horizontal forces."""
    rma, base = aggregate_horizontal(results, force_mags)
    m = np.array(force_mags)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Success Rate
    ax1.plot(m, rma["sr"], "-o", color=RMA_COLOR, label="RMA (encoder active)")
    ax1.plot(m, base["sr"], "--s", color=BASE_COLOR, label="Baseline ($z_t = 0$)")
    ax1.fill_between(m, base["sr"], rma["sr"],
                     where=np.array(rma["sr"]) > np.array(base["sr"]),
                     alpha=0.15, color=RMA_COLOR, interpolate=True)
    ax1.fill_between(m, rma["sr"], base["sr"],
                     where=np.array(base["sr"]) > np.array(rma["sr"]),
                     alpha=0.15, color=BASE_COLOR, interpolate=True)

    # Annotate the key crossover
    ax1.annotate("RMA advantage\n(22% vs 0%)", xy=(100, 22),
                 xytext=(70, 55), fontsize=11,
                 arrowprops=dict(arrowstyle="->", color=RMA_COLOR, lw=1.5),
                 color=RMA_COLOR, fontweight="bold")

    ax1.set_xlabel("Perturbation Force Magnitude (N)")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("(a) Survival Under Horizontal Forces")
    ax1.set_ylim(-5, 108)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Tracking RMSE
    rma_t = np.array(rma["track"])
    base_t = np.array(base["track"])
    rma_ts = np.array(rma["track_std"])
    base_ts = np.array(base["track_std"])

    rma_mask = ~np.isnan(rma_t)
    base_mask = ~np.isnan(base_t)

    if rma_mask.any():
        ax2.errorbar(m[rma_mask], rma_t[rma_mask], yerr=rma_ts[rma_mask],
                     fmt="-o", color=RMA_COLOR, capsize=4, label="RMA")
    if base_mask.any():
        ax2.errorbar(m[base_mask], base_t[base_mask], yerr=base_ts[base_mask],
                     fmt="--s", color=BASE_COLOR, capsize=4, label="Baseline")

    ax2.set_xlabel("Perturbation Force Magnitude (N)")
    ax2.set_ylabel("Tracking RMSE (m/s)")
    ax2.set_title("(b) Velocity Tracking Error (survivors)")
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in (".png", ".pdf"):
        path = os.path.join(OUT_DIR, f"fig_rma_horizontal_comparison{ext}")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()


def fig2_comprehensive(results, force_mags):
    """Figure 2: 2x3 comprehensive metrics (horizontal only)."""
    rma, base = aggregate_horizontal(results, force_mags)
    m = np.array(force_mags)

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))

    def plot_metric(ax, rma_vals, base_vals, ylabel, title, fill=False):
        rv = np.array(rma_vals, dtype=float)
        bv = np.array(base_vals, dtype=float)
        rm = ~np.isnan(rv) & (rv != 0)
        bm = ~np.isnan(bv) & (bv != 0)

        if rm.any():
            ax.plot(m[rm], rv[rm], "-o", color=RMA_COLOR, label="RMA")
        if bm.any():
            ax.plot(m[bm], bv[bm], "--s", color=BASE_COLOR, label="Baseline")
        if fill and rm.any() and bm.any():
            both = rm & bm
            if both.any():
                ax.fill_between(m[both], bv[both], rv[both],
                                where=rv[both] > bv[both],
                                alpha=0.1, color=RMA_COLOR)
        ax.set_xlabel("Force (N)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

    plot_metric(axes[0, 0], rma["sr"], base["sr"],
                "Success Rate (%)", "(a) Survival Rate", fill=True)
    axes[0, 0].set_ylim(-5, 108)

    plot_metric(axes[0, 1], rma["surv"], base["surv"],
                "Mean Survival (s)", "(b) Mean Survival Time")

    plot_metric(axes[0, 2], rma["track"], base["track"],
                "RMSE (m/s)", "(c) Velocity Tracking Error")

    plot_metric(axes[1, 0], rma["orient"], base["orient"],
                "Orient. Error", "(d) Orientation Stability")

    plot_metric(axes[1, 1], rma["energy"], base["energy"],
                "Energy (W)", "(e) Energy Consumption")

    plot_metric(axes[1, 2], rma["smooth"], base["smooth"],
                "Smoothness (Nm)", "(f) Torque Smoothness")

    fig.suptitle("RMA vs Baseline Under Horizontal Torso Perturbation\n"
                 "(H1-2 Humanoid, MuJoCo, 10s trials, 9 directions per magnitude)",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        path = os.path.join(OUT_DIR, f"fig_rma_comprehensive{ext}")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()


def fig3_adaptation_timeseries():
    """Figure 3: Temporal adaptation analysis with time-series data.

    Runs fresh trials at 50N +X with/without encoder, collecting per-step data.
    """
    from evaluate_rma_book import (
        EVAL_CONFIG, load_models, run_trial,
        quat_rotate_inverse, compute_obs, pd_control
    )

    cfg = EVAL_CONFIG

    # Find latest checkpoint
    log_dir = os.path.join(_REPO_ROOT, "logs", "h1_2_rma", "Apr05_19-19-47_")
    pts = sorted([f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")],
                 key=lambda x: int(x.replace("model_", "").replace(".pt", "")))
    ckpt_path = os.path.join(log_dir, pts[-1])
    policy, encoder = load_models(ckpt_path, cfg)
    m = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    m.opt.timestep = cfg["simulation_dt"]

    force_mag = 50.0
    force_vec = np.array([force_mag, 0, 0], dtype=np.float32)

    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.2]})

    for use_enc, label, color, ls in [(True, "RMA", RMA_COLOR, "-"),
                                       (False, "Baseline", BASE_COLOR, "--")]:
        result, ts = run_trial(m, policy, encoder, cfg, force_vec, "torso",
                               use_enc, collect_timeseries=True)
        if not ts:
            continue

        times = [p["t"] for p in ts]
        heights = [p["base_height"] for p in ts]
        orient_errs = [p["orient_err"] for p in ts]
        vxs = [p["vx"] if p["vx"] is not None else float('nan') for p in ts]

        axes[0].plot(times, heights, ls, color=color, label=label, alpha=0.85)
        axes[1].plot(times, orient_errs, ls, color=color, label=label, alpha=0.85)
        axes[2].plot(times, vxs, ls, color=color, label=label, alpha=0.85)

        if use_enc:
            z_ts = np.array([p["z_t"] for p in ts])
            for dim in range(min(z_ts.shape[1], 8)):
                axes[3].plot(times, z_ts[:, dim], alpha=0.7,
                             label=f"$z_{{{dim}}}$")

    force_start = cfg["force_start_time"]
    for i, ax in enumerate(axes):
        ax.axvline(x=force_start, color="gray", linestyle=":", alpha=0.6, linewidth=1.5)
        # Add force onset annotation only on first subplot
        if i == 0:
            ax.annotate("Force onset", xy=(force_start, ax.get_ylim()[1]),
                         xytext=(force_start + 0.3, ax.get_ylim()[1]),
                         fontsize=10, color="gray", va="top")
        # Shade pre-force and post-force regions
        ax.axvspan(0, force_start, alpha=0.04, color="green")
        ax.axvspan(force_start, max(times), alpha=0.04, color="red")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Base Height (m)")
    axes[0].set_title("(a) Base Height Stability")
    axes[0].legend(loc="lower left", framealpha=0.9)

    axes[1].set_ylabel("Orientation Error")
    axes[1].set_title("(b) Body Orientation Deviation")
    axes[1].legend(loc="upper left", framealpha=0.9)

    axes[2].set_ylabel("Forward Velocity (m/s)")
    axes[2].axhline(y=0.5, color="green", linestyle="--", alpha=0.5,
                    linewidth=1.5, label="Cmd = 0.5 m/s")
    axes[2].set_title("(c) Forward Velocity Tracking")
    axes[2].legend(loc="upper right", framealpha=0.9)

    axes[3].set_ylabel("Extrinsics $\\hat{z}_t$")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("(d) Encoder Latent Vector Components (RMA only)")
    axes[3].legend(loc="upper right", ncol=4, fontsize=10, framealpha=0.9)

    fig.suptitle(f"Temporal Adaptation: {force_mag:.0f}N +X Torso Push at t = {force_start}s",
                 fontsize=16, y=1.01)
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        path = os.path.join(OUT_DIR, f"fig_adaptation_timeseries{ext}")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()


def fig4_direction_heatmap(results):
    """Figure 4: Direction × magnitude heatmap showing per-trial outcomes."""
    force_mags = sorted(set(float(r['force_mag']) for r in results if float(r['force_mag']) > 0))
    directions = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for method, ax, title in [("RMA", ax1, "RMA (encoder active)"),
                               ("Baseline", ax2, "Baseline ($z_t = 0$)")]:
        grid = np.full((len(directions), len(force_mags)), np.nan)

        for i, d in enumerate(directions):
            for j, mag in enumerate(force_mags):
                row = [r for r in results if r['method'] == method
                       and float(r['force_mag']) == mag
                       and r['direction'] == d]
                if row:
                    grid[i, j] = float(row[0]['survival_time'])

        im = ax.imshow(grid, aspect='auto', cmap='RdYlGn', vmin=0, vmax=10,
                       interpolation='nearest')
        ax.set_xticks(range(len(force_mags)))
        ax.set_xticklabels([f"{int(m)}" for m in force_mags], rotation=45)
        ax.set_yticks(range(len(directions)))
        ax.set_yticklabels(directions)
        ax.set_xlabel("Force Magnitude (N)")
        ax.set_ylabel("Direction")
        ax.set_title(title)

        # Add text annotations
        for i in range(len(directions)):
            for j in range(len(force_mags)):
                val = grid[i, j]
                if not np.isnan(val):
                    text = "10s" if val >= 9.99 else f"{val:.1f}s"
                    color = "black" if val > 5 else "white"
                    ax.text(j, i, text, ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold" if val >= 9.99 else "normal")

    fig.colorbar(im, ax=[ax1, ax2], label="Survival Time (s)", shrink=0.8)
    fig.suptitle("Per-Direction Survival: RMA vs Baseline\n(Axis-aligned torso forces, 10s trials)",
                 fontsize=15, y=1.05)
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        path = os.path.join(OUT_DIR, f"fig_direction_heatmap{ext}")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()


def generate_latex_horizontal_table(results, force_mags):
    """LaTeX table with horizontal-only metrics."""
    rma, base = aggregate_horizontal(results, force_mags)

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\caption{RMA vs.\ Baseline under horizontal torso force perturbation "
                 r"in MuJoCo. Each magnitude is tested over 9 horizontal directions "
                 r"(4 axis-aligned + 5 random). Vertical ($\pm Z$) directions excluded "
                 r"as they trivially assist/load the robot. Metrics are averaged over surviving trials.}")
    lines.append(r"\label{tab:rma_horizontal}")
    lines.append(r"\begin{tabularx}{\textwidth}{r|cc|cc|cc|cc}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{Force (N)}} & "
                 r"\multicolumn{2}{c|}{\textbf{Success (\%)}} & "
                 r"\multicolumn{2}{c|}{\textbf{RMSE (m/s)}} & "
                 r"\multicolumn{2}{c|}{\textbf{Orient. Err.}} & "
                 r"\multicolumn{2}{c}{\textbf{Energy (W)}} \\")
    lines.append(r"& RMA & Base & RMA & Base & RMA & Base & RMA & Base \\")
    lines.append(r"\midrule")

    for i, mag in enumerate(force_mags):
        def fmt(v, p=3):
            return f"{v:.{p}f}" if not np.isnan(float(v)) else "---"

        sr_r, sr_b = rma["sr"][i], base["sr"][i]
        tr_r, tr_b = rma["track"][i], base["track"][i]
        or_r, or_b = rma["orient"][i], base["orient"][i]
        en_r, en_b = rma["energy"][i], base["energy"][i]

        sr_r_s, sr_b_s = fmt(sr_r, 1), fmt(sr_b, 1)
        tr_r_s, tr_b_s = fmt(tr_r), fmt(tr_b)
        or_r_s, or_b_s = fmt(or_r), fmt(or_b)
        en_r_s, en_b_s = fmt(en_r, 1), fmt(en_b, 1)

        # Bold winner
        def bold(a, b, a_s, b_s, lower=True):
            try:
                va, vb = float(a), float(b)
                if (lower and va < vb) or (not lower and va > vb):
                    return r"\textbf{" + a_s + "}", b_s
                elif (lower and vb < va) or (not lower and vb > va):
                    return a_s, r"\textbf{" + b_s + "}"
            except (ValueError, TypeError):
                pass
            return a_s, b_s

        sr_r_s, sr_b_s = bold(sr_r, sr_b, sr_r_s, sr_b_s, lower=False)
        tr_r_s, tr_b_s = bold(tr_r, tr_b, tr_r_s, tr_b_s, lower=True)
        or_r_s, or_b_s = bold(or_r, or_b, or_r_s, or_b_s, lower=True)
        en_r_s, en_b_s = bold(en_r, en_b, en_r_s, en_b_s, lower=True)

        lines.append(f"{int(mag)} & {sr_r_s} & {sr_b_s} & "
                     f"{tr_r_s} & {tr_b_s} & "
                     f"{or_r_s} & {or_b_s} & "
                     f"{en_r_s} & {en_b_s} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    path = os.path.join(OUT_DIR, "rma_horizontal_table.tex")
    with open(path, "w") as f:
        f.write(tex)
    print(f"  Saved: {path}")
    return tex


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    results = load_results()
    force_mags = [0, 5, 10, 15, 20, 30, 40, 50, 60, 75, 100, 125, 150]

    print("Generating book chapter figures...")
    fig1_main_comparison(results, force_mags)
    fig2_comprehensive(results, force_mags)
    fig3_adaptation_timeseries()
    fig4_direction_heatmap(results)

    print("\nGenerating LaTeX table...")
    tex = generate_latex_horizontal_table(results, force_mags)
    print("\nLaTeX preview:")
    print(tex)

    print(f"\nAll figures saved to: {OUT_DIR}/")
