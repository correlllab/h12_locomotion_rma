"""Comprehensive RMA evaluation — paired RMA vs baseline across commands/forces/seeds.

Goal: produce apples-to-apples evidence that the RMA encoder helps.  For every
(body, magnitude, direction, command, seed) condition we run TWO trials
back-to-back with identical initial state and force:
  1. RMA:      z_t = encoder(e_t_norm)   (encoder sees ground-truth forces)
  2. Baseline: z_t = encoder(zeros)      (no force info)
Paired survival and tracking RMSE let you directly measure the RMA lift per
condition instead of comparing unrelated averages.

Outputs land in a clearly-labeled directory:
  eval_results/<ckpt_stem>__<tag>__<timestamp>/
    ├── paired_results.csv
    ├── summary.md
    ├── success_rate_vs_force.png
    ├── tracking_rmse_vs_force.png
    └── rma_lift_heatmap.png

Usage
-----
  python MujocoDeploy/comprehensive_eval.py \\
      --ckpt logs/h1_2_rma/Apr16_15-12-01_/model_6000.pt \\
      --tag phase1_curriculum \\
      --seeds 3

  # Re-plot / re-summarise only (no new sims):
  python MujocoDeploy/comprehensive_eval.py \\
      --results_dir eval_results/model_6000__phase1_curriculum__2026-04-17_17-55 \\
      --summary_only
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mujoco

from rsl_rl.modules import ActorCriticRecurrent
from rma.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg

# Reuse helpers from sweep_rma_forces (same sim setup)
from MujocoDeploy.sweep_rma_forces import (
    AXIS_DIRECTIONS,
    BODY_KEY_MAP,
    _remap_state_dict,
    compute_obs,
    load_models,
    normalize_et_np,
    pd_control,
    quat_rotate_inverse,
    sample_sphere,
)


# ──────────────────────────────────────────────────────────────
#  Trial spec + result
# ──────────────────────────────────────────────────────────────

@dataclass
class Condition:
    """A force+command+seed specification. Both RMA and baseline are
    evaluated under this exact condition for a paired comparison."""
    body: str                     # "torso" | "left_wrist" | "right_wrist" | "combined"
    magnitude_tag: str            # e.g. "10N" or "10/5/5N"
    direction_tag: str            # "+X" | "-Z" | "sph3"
    torso_force: np.ndarray       # (3,)
    left_wrist_force: np.ndarray  # (3,)
    right_wrist_force: np.ndarray # (3,)
    command: np.ndarray           # (3,) vx, vy, yaw
    command_tag: str              # "walk_slow" | "stand" etc.
    seed: int


@dataclass
class TrialResult:
    condition: Condition
    use_encoder: bool
    survival_time: float
    success: bool
    tracking_rmse_vx: float
    tracking_rmse_vy: float
    tracking_rmse_xy: float
    mean_orientation_err: float
    mean_action_magnitude: float  # diagnostic: policy "effort"


# ──────────────────────────────────────────────────────────────
#  Single trial runner (uses per-seed deterministic noise)
# ──────────────────────────────────────────────────────────────

def run_trial(
    m_template,
    policy,
    encoder,
    cfg,
    condition: Condition,
    use_encoder: bool,
) -> TrialResult:
    """Run one headless MuJoCo trial with deterministic init."""
    eval_cfg = cfg["eval"]
    dt = cfg["simulation_dt"]
    decim = cfg["control_decimation"]
    n_leg = cfg["num_actions"]
    n_steps = int(eval_cfg["duration"] / dt)
    fall_height = eval_cfg["fall_height"]
    force_start = eval_cfg.get("force_start_time", 1.0)
    tracking_warmup = eval_cfg.get("tracking_warmup", 2.0)
    phase_period = cfg.get("phase_period", 0.8)
    max_tau = 300.0

    # Fresh sim data, seeded init perturbation on default pose
    d = mujoco.MjData(m_template)
    rng = np.random.default_rng(condition.seed)
    n_joints = d.qpos.shape[0] - 7

    # Small deterministic init jitter on legs (so different seeds really
    # test different initial conditions, not just encoder noise)
    default_legs = np.array(cfg["default_angles"][:n_leg], dtype=np.float32)
    default_arms = np.array(cfg.get("default_angles_arms",
                                     np.zeros(n_joints - n_leg)), dtype=np.float32)
    jitter = rng.normal(0, 0.02, size=n_leg).astype(np.float32)
    d.qpos[7:7+n_leg] = default_legs + jitter
    if n_joints > n_leg:
        d.qpos[7+n_leg:7+n_joints] = default_arms[:n_joints - n_leg]

    # Body IDs
    torso_id = mujoco.mj_name2id(m_template, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    left_id = mujoco.mj_name2id(m_template, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
    right_id = mujoco.mj_name2id(m_template, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")

    # Upper-body PD
    kps_arm = np.array(cfg.get("kps_arms",
                                np.ones(n_joints - n_leg) * 100), dtype=np.float32)
    kds_arm = np.array(cfg.get("kds_arms",
                                np.ones(n_joints - n_leg) * 5), dtype=np.float32)

    kps_leg = np.array(cfg["kps"], dtype=np.float32)
    kds_leg = np.array(cfg["kds"], dtype=np.float32)

    # Reset LSTM hidden state per trial (fair comparison)
    policy.memory_a.hidden_states = None

    action = np.zeros(n_leg, dtype=np.float32)
    counter = 0
    survival_time = eval_cfg["duration"]

    vx_errors, vy_errors, orient_errors, action_mags = [], [], [], []

    cmd = condition.command.astype(np.float32)
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)

    for step in range(n_steps):
        t = step * dt

        # External forces only after stabilisation
        d.xfrc_applied[:] = 0
        if t >= force_start:
            if torso_id >= 0:
                d.xfrc_applied[torso_id, :3] = condition.torso_force
            if left_id >= 0:
                d.xfrc_applied[left_id, :3] = condition.left_wrist_force
            if right_id >= 0:
                d.xfrc_applied[right_id, :3] = condition.right_wrist_force

        # PD control (legs)
        target_dof = action * cfg["action_scale"] + default_legs
        leg_tau = pd_control(target_dof, d.qpos[7:7+n_leg], kps_leg,
                             d.qvel[6:6+n_leg], kds_leg)
        leg_tau = np.clip(np.nan_to_num(leg_tau), -max_tau, max_tau)
        d.ctrl[:n_leg] = leg_tau

        # PD upper body
        if n_joints > n_leg and d.ctrl.shape[0] > n_leg:
            n_upper = min(n_joints - n_leg, d.ctrl.shape[0] - n_leg)
            arm_tau = pd_control(default_arms[:n_upper],
                                 d.qpos[7+n_leg:7+n_leg+n_upper],
                                 kps_arm[:n_upper],
                                 d.qvel[6+n_leg:6+n_leg+n_upper],
                                 kds_arm[:n_upper])
            arm_tau = np.clip(np.nan_to_num(arm_tau), -max_tau, max_tau)
            d.ctrl[n_leg:n_leg+n_upper] = arm_tau

        mujoco.mj_step(m_template, d)
        counter += 1

        # Fall check
        if d.qpos[2] < fall_height:
            survival_time = t
            break

        # Policy step
        if counter % decim == 0:
            phase = (t / phase_period) % 1.0
            obs_47, proj_grav = compute_obs(d, cfg, action, cmd, phase, n_leg)

            if use_encoder and t >= force_start:
                e_t = np.concatenate([condition.torso_force,
                                      condition.left_wrist_force,
                                      condition.right_wrist_force]).astype(np.float32)
            else:
                e_t = np.zeros(cfg.get("rma_et_dim", 9), dtype=np.float32)

            e_t_norm = normalize_et_np(e_t)
            with torch.no_grad():
                z_t = encoder(torch.from_numpy(e_t_norm).unsqueeze(0).float()).numpy().squeeze()

            actor_obs = np.concatenate([obs_47, z_t]).astype(np.float32)
            with torch.no_grad():
                action = policy.act_inference(
                    torch.from_numpy(actor_obs).unsqueeze(0).float()
                ).numpy().squeeze()

            if t >= tracking_warmup:
                quat = d.qpos[3:7].copy()
                base_vel_body = quat_rotate_inverse(quat, d.qvel[0:3].copy())
                vx_errors.append((cmd[0] - base_vel_body[0]) ** 2)
                vy_errors.append((cmd[1] - base_vel_body[1]) ** 2)
                orient_errors.append(np.linalg.norm(proj_grav[:2]))
                action_mags.append(np.linalg.norm(action))

    success = survival_time >= eval_cfg["duration"] - 0.01
    rmse_vx = float(np.sqrt(np.mean(vx_errors))) if vx_errors else float("nan")
    rmse_vy = float(np.sqrt(np.mean(vy_errors))) if vy_errors else float("nan")
    rmse_xy = (float(np.sqrt(np.mean(np.array(vx_errors) + np.array(vy_errors))))
               if vx_errors else float("nan"))
    mean_orient = float(np.mean(orient_errors)) if orient_errors else float("nan")
    mean_act = float(np.mean(action_mags)) if action_mags else float("nan")

    return TrialResult(
        condition=condition,
        use_encoder=use_encoder,
        survival_time=round(survival_time, 3),
        success=success,
        tracking_rmse_vx=round(rmse_vx, 4),
        tracking_rmse_vy=round(rmse_vy, 4),
        tracking_rmse_xy=round(rmse_xy, 4),
        mean_orientation_err=round(mean_orient, 4),
        mean_action_magnitude=round(mean_act, 4),
    )


# ──────────────────────────────────────────────────────────────
#  Condition generation
# ──────────────────────────────────────────────────────────────

COMMANDS: Dict[str, np.ndarray] = {
    "stand":        np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "walk_slow":    np.array([0.3, 0.0, 0.0], dtype=np.float32),
    "walk":         np.array([0.5, 0.0, 0.0], dtype=np.float32),
    "walk_fast":    np.array([0.8, 0.0, 0.0], dtype=np.float32),
    "side_left":    np.array([0.0, 0.3, 0.0], dtype=np.float32),
    "side_right":   np.array([0.0, -0.3, 0.0], dtype=np.float32),
    "turn_left":    np.array([0.3, 0.0, 0.5], dtype=np.float32),
}


def generate_conditions(
    magnitudes: List[float],
    n_spherical: int,
    combined_grid: List[List[float]],
    commands: List[str],
    seeds: List[int],
    sph_seed: int,
    bodies: Optional[List[str]] = None,
) -> List[Condition]:
    """Build the full grid of conditions (each will be run RMA + baseline)."""
    conditions: List[Condition] = []
    zero = np.zeros(3, dtype=np.float32)
    rng = np.random.default_rng(sph_seed)
    bodies = bodies or ["torso", "left_wrist", "right_wrist"]

    # --- Single-body sweeps: axis + spherical ---
    for body in bodies:
        for mag in magnitudes:
            if mag == 0:
                for cmd_tag in commands:
                    for sd in seeds:
                        conditions.append(Condition(
                            body=body,
                            magnitude_tag="0N",
                            direction_tag="zero",
                            torso_force=zero.copy(),
                            left_wrist_force=zero.copy(),
                            right_wrist_force=zero.copy(),
                            command=COMMANDS[cmd_tag].copy(),
                            command_tag=cmd_tag,
                            seed=sd,
                        ))
                continue

            # Axis directions
            for dname, dvec in AXIS_DIRECTIONS.items():
                for cmd_tag in commands:
                    for sd in seeds:
                        force = dvec * mag
                        kwargs = {k: zero.copy() for k in BODY_KEY_MAP.values()}
                        kwargs[BODY_KEY_MAP[body]] = force
                        conditions.append(Condition(
                            body=body,
                            magnitude_tag=f"{mag:g}N",
                            direction_tag=dname,
                            command=COMMANDS[cmd_tag].copy(),
                            command_tag=cmd_tag,
                            seed=sd,
                            **kwargs,
                        ))

            # Spherical random directions
            sph_dirs = sample_sphere(n_spherical, rng)
            for i, dvec in enumerate(sph_dirs):
                for cmd_tag in commands:
                    for sd in seeds:
                        force = dvec.astype(np.float32) * mag
                        kwargs = {k: zero.copy() for k in BODY_KEY_MAP.values()}
                        kwargs[BODY_KEY_MAP[body]] = force
                        conditions.append(Condition(
                            body=body,
                            magnitude_tag=f"{mag:g}N",
                            direction_tag=f"sph{i}",
                            command=COMMANDS[cmd_tag].copy(),
                            command_tag=cmd_tag,
                            seed=sd,
                            **kwargs,
                        ))

    # --- Combined sweeps: forces on all 3 bodies at once ---
    for combo in combined_grid:
        t_mag, l_mag, r_mag = combo
        mag_tag = f"{t_mag:g}/{l_mag:g}/{r_mag:g}N"
        for i in range(n_spherical):
            dirs = sample_sphere(3, rng)
            for cmd_tag in commands:
                for sd in seeds:
                    conditions.append(Condition(
                        body="combined",
                        magnitude_tag=mag_tag,
                        direction_tag=f"sph{i}",
                        torso_force=(dirs[0] * t_mag).astype(np.float32),
                        left_wrist_force=(dirs[1] * l_mag).astype(np.float32),
                        right_wrist_force=(dirs[2] * r_mag).astype(np.float32),
                        command=COMMANDS[cmd_tag].copy(),
                        command_tag=cmd_tag,
                        seed=sd,
                    ))

    return conditions


# ──────────────────────────────────────────────────────────────
#  CSV
# ──────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "body", "magnitude_tag", "direction_tag", "command_tag", "seed",
    "torso_fx", "torso_fy", "torso_fz",
    "left_fx", "left_fy", "left_fz",
    "right_fx", "right_fy", "right_fz",
    "total_force_mag",
    "use_encoder",
    "survival_time", "success",
    "tracking_rmse_vx", "tracking_rmse_vy", "tracking_rmse_xy",
    "mean_orientation_err", "mean_action_magnitude",
]


def _row(r: TrialResult) -> dict:
    c = r.condition
    total = (float(np.linalg.norm(c.torso_force))
             + float(np.linalg.norm(c.left_wrist_force))
             + float(np.linalg.norm(c.right_wrist_force)))
    return {
        "body": c.body,
        "magnitude_tag": c.magnitude_tag,
        "direction_tag": c.direction_tag,
        "command_tag": c.command_tag,
        "seed": c.seed,
        "torso_fx": c.torso_force[0], "torso_fy": c.torso_force[1], "torso_fz": c.torso_force[2],
        "left_fx": c.left_wrist_force[0], "left_fy": c.left_wrist_force[1], "left_fz": c.left_wrist_force[2],
        "right_fx": c.right_wrist_force[0], "right_fy": c.right_wrist_force[1], "right_fz": c.right_wrist_force[2],
        "total_force_mag": round(total, 3),
        "use_encoder": r.use_encoder,
        "survival_time": r.survival_time,
        "success": r.success,
        "tracking_rmse_vx": r.tracking_rmse_vx,
        "tracking_rmse_vy": r.tracking_rmse_vy,
        "tracking_rmse_xy": r.tracking_rmse_xy,
        "mean_orientation_err": r.mean_orientation_err,
        "mean_action_magnitude": r.mean_action_magnitude,
    }


def write_csv(results: List[TrialResult], path: str):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow(_row(r))


def read_csv(path: str) -> List[dict]:
    with open(path, "r") as f:
        return list(csv.DictReader(f))


# ──────────────────────────────────────────────────────────────
#  Summary + plots
# ──────────────────────────────────────────────────────────────

def _paired_index(rows: List[dict]) -> Dict[Tuple, Dict[str, dict]]:
    """Index rows by (body, mag, dir, cmd, seed) → {'rma': row, 'base': row}."""
    idx: Dict[Tuple, Dict[str, dict]] = {}
    for r in rows:
        key = (r["body"], r["magnitude_tag"], r["direction_tag"],
               r["command_tag"], r["seed"])
        side = "rma" if r["use_encoder"] == "True" else "base"
        idx.setdefault(key, {})[side] = r
    return idx


def write_summary(rows: List[dict], out_path: str):
    idx = _paired_index(rows)
    paired = [v for v in idx.values() if "rma" in v and "base" in v]
    n_pairs = len(paired)

    def _success_rate(side: str, filter_fn=lambda r: True) -> Tuple[int, int]:
        subset = [p[side] for p in paired if filter_fn(p[side])]
        if not subset:
            return 0, 0
        succ = sum(1 for r in subset if r["success"] == "True")
        return succ, len(subset)

    lines: List[str] = []
    lines.append("# RMA Comprehensive Evaluation — Summary\n")
    lines.append(f"- Paired trials: **{n_pairs}** (each run twice: RMA + baseline)\n")

    rma_s, rma_t = _success_rate("rma")
    bas_s, bas_t = _success_rate("base")
    lines.append(f"- Overall survival: **RMA** {rma_s}/{rma_t} ({100*rma_s/max(rma_t,1):.1f}%) "
                 f"vs **baseline** {bas_s}/{bas_t} ({100*bas_s/max(bas_t,1):.1f}%)\n")

    # Paired lift: fraction of pairs where RMA survived AND baseline fell
    both = [(p["rma"]["success"], p["base"]["success"]) for p in paired]
    rma_only = sum(1 for a, b in both if a == "True" and b != "True")
    base_only = sum(1 for a, b in both if a != "True" and b == "True")
    both_yes = sum(1 for a, b in both if a == "True" and b == "True")
    both_no = sum(1 for a, b in both if a != "True" and b != "True")
    lines.append("\n## Paired survival breakdown\n")
    lines.append(f"- Both survived: **{both_yes}** ({100*both_yes/max(n_pairs,1):.1f}%)")
    lines.append(f"- RMA only: **{rma_only}** ({100*rma_only/max(n_pairs,1):.1f}%)   ← RMA lift")
    lines.append(f"- Baseline only: **{base_only}** ({100*base_only/max(n_pairs,1):.1f}%)   ← RMA regression")
    lines.append(f"- Both fell: **{both_no}** ({100*both_no/max(n_pairs,1):.1f}%)\n")

    # Per-body, per-magnitude table
    lines.append("\n## Per-body × per-magnitude survival rate (paired)\n")
    lines.append("| Body | Magnitude | RMA % | Baseline % | Pairs | RMA-only | Base-only |")
    lines.append("|------|-----------|-------|------------|-------|----------|-----------|")
    # Group
    groups: Dict[Tuple[str, str], List[Dict[str, dict]]] = {}
    for p in paired:
        key = (p["rma"]["body"], p["rma"]["magnitude_tag"])
        groups.setdefault(key, []).append(p)
    for (body, mag), group in sorted(groups.items(),
                                     key=lambda kv: (kv[0][0], _mag_key(kv[0][1]))):
        n = len(group)
        rma_succ = sum(1 for g in group if g["rma"]["success"] == "True")
        base_succ = sum(1 for g in group if g["base"]["success"] == "True")
        rma_only = sum(1 for g in group
                       if g["rma"]["success"] == "True" and g["base"]["success"] != "True")
        base_only = sum(1 for g in group
                        if g["rma"]["success"] != "True" and g["base"]["success"] == "True")
        lines.append(f"| {body} | {mag} | {100*rma_succ/n:.1f} | {100*base_succ/n:.1f} "
                     f"| {n} | {rma_only} | {base_only} |")

    # Per-command table
    lines.append("\n## Per-command survival rate (paired)\n")
    lines.append("| Command | RMA % | Baseline % | Pairs |")
    lines.append("|---------|-------|------------|-------|")
    cmd_groups: Dict[str, List[Dict[str, dict]]] = {}
    for p in paired:
        cmd_groups.setdefault(p["rma"]["command_tag"], []).append(p)
    for cmd, group in sorted(cmd_groups.items()):
        n = len(group)
        rma_succ = sum(1 for g in group if g["rma"]["success"] == "True")
        base_succ = sum(1 for g in group if g["base"]["success"] == "True")
        lines.append(f"| {cmd} | {100*rma_succ/n:.1f} | {100*base_succ/n:.1f} | {n} |")

    # Tracking RMSE (survivors only, paired)
    lines.append("\n## Tracking RMSE on common survivors (mean ± std, m/s)\n")
    lines.append("| Body | Magnitude | RMA | Baseline | Pairs |")
    lines.append("|------|-----------|-----|----------|-------|")
    for (body, mag), group in sorted(groups.items(),
                                     key=lambda kv: (kv[0][0], _mag_key(kv[0][1]))):
        common = [g for g in group
                  if g["rma"]["success"] == "True" and g["base"]["success"] == "True"]
        if not common:
            continue
        rma_track = [float(g["rma"]["tracking_rmse_xy"]) for g in common
                     if g["rma"]["tracking_rmse_xy"] not in ("nan", "")]
        base_track = [float(g["base"]["tracking_rmse_xy"]) for g in common
                      if g["base"]["tracking_rmse_xy"] not in ("nan", "")]
        if not rma_track or not base_track:
            continue
        lines.append(f"| {body} | {mag} | {np.mean(rma_track):.4f}±{np.std(rma_track):.4f} "
                     f"| {np.mean(base_track):.4f}±{np.std(base_track):.4f} "
                     f"| {len(common)} |")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))


def _mag_key(s: str) -> float:
    """Sort key for magnitude tags: '10N' → 10, '10/5/5N' → 20."""
    s = s.rstrip("N")
    try:
        return sum(float(x) for x in s.split("/"))
    except ValueError:
        return 0.0


def make_plots(rows: List[dict], out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    idx = _paired_index(rows)
    paired = [v for v in idx.values() if "rma" in v and "base" in v]

    # ── Plot 1: survival rate vs magnitude per body ────────────────────
    bodies = ["torso", "left_wrist", "right_wrist"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, body in zip(axes, bodies):
        body_pairs = [p for p in paired if p["rma"]["body"] == body]
        mag_groups: Dict[float, List[Dict[str, dict]]] = {}
        for p in body_pairs:
            mag = _mag_key(p["rma"]["magnitude_tag"])
            mag_groups.setdefault(mag, []).append(p)
        if not mag_groups:
            ax.set_title(f"{body} (no data)")
            continue
        mags = sorted(mag_groups.keys())
        rma_rate = [100 * sum(1 for g in mag_groups[m] if g["rma"]["success"] == "True")
                    / len(mag_groups[m]) for m in mags]
        base_rate = [100 * sum(1 for g in mag_groups[m] if g["base"]["success"] == "True")
                     / len(mag_groups[m]) for m in mags]
        ax.plot(mags, rma_rate, "-o", color="C0", label="RMA", markersize=6)
        ax.plot(mags, base_rate, "--s", color="C3", label="Baseline", markersize=6)
        ax.set_xlabel("Force magnitude (N)")
        ax.set_title(body)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Survival rate (%)")
    fig.suptitle("Paired RMA vs baseline — survival rate vs force")
    plt.tight_layout()
    p = os.path.join(out_dir, "success_rate_vs_force.png")
    plt.savefig(p, dpi=150)
    print(f"  Saved: {p}")
    plt.close()

    # ── Plot 2: tracking RMSE vs magnitude per body (common survivors) ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, body in zip(axes, bodies):
        body_pairs = [p for p in paired if p["rma"]["body"] == body]
        mag_groups = {}
        for p in body_pairs:
            mag = _mag_key(p["rma"]["magnitude_tag"])
            mag_groups.setdefault(mag, []).append(p)
        if not mag_groups:
            ax.set_title(f"{body} (no data)")
            continue
        mags = sorted(mag_groups.keys())
        rma_means, rma_stds, base_means, base_stds = [], [], [], []
        for m in mags:
            common = [g for g in mag_groups[m]
                      if g["rma"]["success"] == "True" and g["base"]["success"] == "True"]
            if not common:
                rma_means.append(np.nan); rma_stds.append(0)
                base_means.append(np.nan); base_stds.append(0)
                continue
            rv = [float(g["rma"]["tracking_rmse_xy"]) for g in common
                  if g["rma"]["tracking_rmse_xy"] not in ("nan", "")]
            bv = [float(g["base"]["tracking_rmse_xy"]) for g in common
                  if g["base"]["tracking_rmse_xy"] not in ("nan", "")]
            rma_means.append(np.mean(rv) if rv else np.nan)
            rma_stds.append(np.std(rv) if rv else 0)
            base_means.append(np.mean(bv) if bv else np.nan)
            base_stds.append(np.std(bv) if bv else 0)
        ax.errorbar(mags, rma_means, yerr=rma_stds, fmt="-o", color="C0",
                    capsize=3, label="RMA")
        ax.errorbar(mags, base_means, yerr=base_stds, fmt="--s", color="C3",
                    capsize=3, label="Baseline")
        ax.set_xlabel("Force magnitude (N)")
        ax.set_title(body)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Tracking RMSE (m/s)")
    fig.suptitle("Paired tracking RMSE on common survivors")
    plt.tight_layout()
    p = os.path.join(out_dir, "tracking_rmse_vs_force.png")
    plt.savefig(p, dpi=150)
    print(f"  Saved: {p}")
    plt.close()

    # ── Plot 3: RMA lift heatmap (body × magnitude) ────────────────────
    rows_labels = bodies + ["combined"]
    mag_tags_set: set = set()
    for p in paired:
        mag_tags_set.add(p["rma"]["magnitude_tag"])
    mag_tags = sorted(mag_tags_set, key=_mag_key)
    lift_matrix = np.zeros((len(rows_labels), len(mag_tags)))
    count_matrix = np.zeros_like(lift_matrix)
    for p in paired:
        body = p["rma"]["body"]
        if body not in rows_labels:
            continue
        i = rows_labels.index(body)
        j = mag_tags.index(p["rma"]["magnitude_tag"])
        rma_ok = p["rma"]["success"] == "True"
        base_ok = p["base"]["success"] == "True"
        if rma_ok and not base_ok:
            lift_matrix[i, j] += 1
        elif base_ok and not rma_ok:
            lift_matrix[i, j] -= 1
        count_matrix[i, j] += 1
    norm_lift = np.where(count_matrix > 0, lift_matrix / np.maximum(count_matrix, 1), 0)

    fig, ax = plt.subplots(figsize=(max(8, len(mag_tags) * 0.8), 4))
    vmax = max(0.3, np.abs(norm_lift).max())
    im = ax.imshow(norm_lift, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(mag_tags)))
    ax.set_xticklabels(mag_tags, rotation=45, ha="right")
    ax.set_yticks(range(len(rows_labels)))
    ax.set_yticklabels(rows_labels)
    for i in range(len(rows_labels)):
        for j in range(len(mag_tags)):
            if count_matrix[i, j] > 0:
                ax.text(j, i, f"{norm_lift[i,j]:+.2f}\nn={int(count_matrix[i,j])}",
                        ha="center", va="center",
                        color="black" if abs(norm_lift[i,j]) < 0.5 * vmax else "white",
                        fontsize=7)
    fig.colorbar(im, ax=ax, label="RMA-only − baseline-only (per pair)")
    ax.set_title("RMA paired lift — positive = RMA helps")
    plt.tight_layout()
    p = os.path.join(out_dir, "rma_lift_heatmap.png")
    plt.savefig(p, dpi=150)
    print(f"  Saved: {p}")
    plt.close()


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

DEFAULT_MAGNITUDES = [0, 5, 10, 15, 20, 25, 30, 40]
DEFAULT_COMBINED = [[5, 5, 5], [10, 5, 5], [15, 10, 10], [20, 15, 15]]
DEFAULT_COMMANDS = ["walk", "walk_fast", "side_left", "turn_left"]

# Presets so you don't have to hand-tune flags.
# Estimated wall times use ~0.15x realtime from sweep_rma_forces observations.
PRESETS: Dict[str, Dict] = {
    "quick": dict(
        magnitudes=[0, 10, 20, 30],
        commands=["walk"],
        seeds=1,
        n_spherical=2,
        combined_grid=[[10, 5, 5]],
    ),
    "standard": dict(
        magnitudes=[0, 5, 10, 15, 20, 30],
        commands=["walk", "side_left"],
        seeds=2,
        n_spherical=2,
        combined_grid=[[10, 5, 5], [15, 10, 10]],
    ),
    "full": dict(
        magnitudes=DEFAULT_MAGNITUDES,
        commands=DEFAULT_COMMANDS,
        seeds=3,
        n_spherical=4,
        combined_grid=DEFAULT_COMBINED,
    ),
    # Push magnitudes well past the training curriculum (60N at 6k/10k,
    # 100N after full curriculum) to find where RMA decisively beats
    # baseline or where both collapse.  Coarse directional coverage,
    # but bigger ranges on both single-body and combined.
    "stress": dict(
        magnitudes=[0, 30, 50, 75, 100, 150, 200],
        commands=["walk", "side_left"],
        seeds=2,
        n_spherical=3,
        combined_grid=[[30, 20, 20], [50, 30, 30], [75, 50, 50], [100, 75, 75]],
    ),
    # Extreme single-body torso stress — isolates torso robustness curve
    # with dense magnitude sampling at the cliff edge.  Fastest way to
    # find the exact survival threshold for RMA vs baseline.
    "stress_torso": dict(
        magnitudes=[0, 40, 60, 80, 100, 120, 150, 180, 220, 260],
        commands=["walk"],
        seeds=3,
        n_spherical=2,
        combined_grid=[],
        bodies=["torso"],
    ),
}


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive paired RMA evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=str,
                        help="Training checkpoint (.pt). Required unless --summary_only.")
    parser.add_argument("--config", type=str,
                        default=os.path.join(_SCRIPT_DIR, "sweep_config.yaml"),
                        help="YAML with sim/PD/xml paths (reuses sweep_config.yaml).")
    parser.add_argument("--tag", type=str, default="run",
                        help="Label appended to the output directory.")
    parser.add_argument("--out_root", type=str,
                        default=os.path.join(_SCRIPT_DIR, "eval_results"),
                        help="Parent directory; the script creates a sub-dir under it.")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Override output dir (also required for --summary_only).")
    parser.add_argument("--preset", type=str, default="standard",
                        choices=list(PRESETS.keys()),
                        help="Preset grid size. Flags below override preset values.")
    parser.add_argument("--magnitudes", type=float, nargs="+", default=None,
                        help="Force magnitudes (N) for single-body sweeps.")
    parser.add_argument("--commands", type=str, nargs="+", default=None,
                        choices=list(COMMANDS.keys()),
                        help="Velocity commands to test.")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Number of seeds per condition.")
    parser.add_argument("--n_spherical", type=int, default=None,
                        help="Random directions per magnitude per body.")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Trial duration (s).")
    parser.add_argument("--summary_only", action="store_true",
                        help="Re-summarise and re-plot from existing results.csv.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print trial count and exit.")
    args = parser.parse_args()

    # ---- Load sim config ----
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    for key in ("policy_path", "encoder_path", "xml_path"):
        if key in cfg and cfg[key] and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.normpath(os.path.join(config_dir, cfg[key]))
    for key in ("kps", "kds", "kps_arms", "kds_arms", "default_angles",
                "default_angles_arms", "cmd_scale"):
        if key in cfg:
            cfg[key] = np.array(cfg[key], dtype=np.float32)
    cfg.setdefault("eval", {})
    cfg["eval"]["duration"] = args.duration
    cfg["eval"].setdefault("fall_height", 0.5)
    cfg["eval"].setdefault("force_start_time", 1.0)
    cfg["eval"].setdefault("tracking_warmup", 2.0)

    # ---- Resolve output directory ----
    if args.results_dir:
        out_dir = args.results_dir
    else:
        if args.ckpt is None:
            parser.error("--ckpt is required unless --results_dir is provided.")
        ckpt_stem = os.path.splitext(os.path.basename(args.ckpt))[0]
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        out_dir = os.path.join(args.out_root, f"{ckpt_stem}__{args.tag}__{ts}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "paired_results.csv")
    summary_path = os.path.join(out_dir, "summary.md")
    print(f"Output directory: {out_dir}")

    # ---- Summary-only path ----
    if args.summary_only:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No paired_results.csv at {csv_path}")
        rows = read_csv(csv_path)
        write_summary(rows, summary_path)
        make_plots(rows, out_dir)
        return

    # ---- Resolve grid: preset + per-flag overrides ----
    preset = PRESETS[args.preset]
    magnitudes = args.magnitudes if args.magnitudes is not None else preset["magnitudes"]
    commands = args.commands if args.commands is not None else preset["commands"]
    n_seeds = args.seeds if args.seeds is not None else preset["seeds"]
    n_spherical = args.n_spherical if args.n_spherical is not None else preset["n_spherical"]
    combined_grid = preset["combined_grid"]
    bodies = preset.get("bodies", ["torso", "left_wrist", "right_wrist"])

    seeds = list(range(n_seeds))
    conditions = generate_conditions(
        magnitudes=magnitudes,
        n_spherical=n_spherical,
        combined_grid=combined_grid,
        commands=commands,
        seeds=seeds,
        sph_seed=42,
        bodies=bodies,
    )
    n_trials = 2 * len(conditions)  # RMA + baseline for each
    est_time_s = n_trials * args.duration * 0.15
    print(f"Preset: {args.preset}")
    print(f"Bodies: {bodies}")
    print(f"Magnitudes (N): {magnitudes}")
    print(f"Commands: {commands}")
    print(f"Seeds per condition: {n_seeds}")
    print(f"Spherical dirs per body×magnitude: {n_spherical}")
    print(f"Conditions: {len(conditions)}  |  trials (paired): {n_trials}")
    print(f"Estimated wall time: ~{est_time_s/60:.1f} min")
    if args.dry_run:
        return

    # ---- Load models ----
    ckpt_path = args.ckpt
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.abspath(ckpt_path)
    policy, encoder = load_models(cfg, ckpt_path)
    print(f"Loaded checkpoint: {ckpt_path}")

    # ---- MuJoCo model ----
    m = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    m.opt.timestep = cfg["simulation_dt"]

    # ---- Run paired trials ----
    results: List[TrialResult] = []
    t_start = time.time()
    for i, cond in enumerate(conditions):
        for use_enc in (True, False):
            r = run_trial(m, policy, encoder, cfg, cond, use_enc)
            results.append(r)
        # Log every 20 pairs
        if (i + 1) % 20 == 0 or (i + 1) == len(conditions):
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(conditions) - i - 1)
            pair = results[-2:]
            tag = f"{cond.body}|{cond.magnitude_tag}|{cond.direction_tag}|{cond.command_tag}|s{cond.seed}"
            print(f"  [{i+1:4d}/{len(conditions)}] "
                  f"RMA={'OK' if pair[0].success else f'F@{pair[0].survival_time:.1f}s':>8s}  "
                  f"base={'OK' if pair[1].success else f'F@{pair[1].survival_time:.1f}s':>8s}  "
                  f"{tag}   [{elapsed:.0f}s, ~{eta:.0f}s left]")

    # ---- Write CSV + summary + plots ----
    write_csv(results, csv_path)
    print(f"\nCSV: {csv_path}")
    rows = read_csv(csv_path)
    write_summary(rows, summary_path)
    make_plots(rows, out_dir)
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
