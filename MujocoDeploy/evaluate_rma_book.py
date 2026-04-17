#!/usr/bin/env python3
"""Rigorous RMA vs Baseline evaluation for book chapter.

Compares RMA (with encoder) against baseline (no encoder, z_t=0) across a
wide force sweep, collecting rich metrics: survival, velocity tracking,
energy, torque, smoothness, orientation error, and time-series data.

Designed to produce publication-quality figures and LaTeX tables.

Usage:
  python evaluate_rma_book.py \
      --ckpt ../logs/h1_2_rma/Apr05_19-19-47_/model_9950.pt
  python evaluate_rma_book.py --plot_only   # re-plot from existing CSV
"""

import os
import sys
import argparse
import csv
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mujoco

from rsl_rl.modules import ActorCriticRecurrent
from rma.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg
from rma.env_factor_spec import FORCE_COMPONENT_RANGE


def normalize_et_np(et_np):
    """Normalize raw force (N) to [-1, 1]."""
    lo, hi = FORCE_COMPONENT_RANGE
    return 2.0 * (et_np - lo) / (hi - lo) - 1.0

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

EVAL_CONFIG = dict(
    # Simulation
    simulation_dt=0.0025,
    control_decimation=8,
    num_actions=12,
    rma_et_dim=9,
    rma_latent_dim=8,
    phase_period=0.8,
    action_scale=0.25,
    ang_vel_scale=0.25,
    dof_pos_scale=1.0,
    dof_vel_scale=0.05,
    cmd_scale=np.array([2.0, 2.0, 0.25], dtype=np.float32),

    # PD gains (legs)
    kps=np.array([200, 200, 200, 300, 40, 40,
                  200, 200, 200, 300, 40, 40], dtype=np.float32),
    kds=np.array([2.5, 2.5, 2.5, 4.0, 2.0, 2.0,
                  2.5, 2.5, 2.5, 4.0, 2.0, 2.0], dtype=np.float32),

    # PD gains (upper body)
    kps_arms=np.array([200,
                       50, 50, 50, 50, 20, 20, 20,
                       50, 50, 50, 50, 20, 20, 20], dtype=np.float32),
    kds_arms=np.array([5.0,
                       2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0,
                       2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float32),

    # Default angles (legs)
    default_angles=np.array([0.0, -0.16, 0.0, 0.36, -0.2, 0.0,
                             0.0, -0.16, 0.0, 0.36, -0.2, 0.0], dtype=np.float32),

    # Default angles (upper body)
    default_angles_arms=np.array([0.0,
                                  0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0,
                                  0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float32),

    # Evaluation
    eval_duration=10.0,           # seconds per trial
    fall_height=0.5,              # base height threshold (m)
    force_start_time=2.0,         # apply forces after stabilization (s)
    tracking_warmup=3.0,          # ignore tracking before this time (s)
    cmd=np.array([0.5, 0.0, 0.0], dtype=np.float32),  # forward walk
    max_tau=300.0,

    # Paths
    xml_path=os.path.join(_SCRIPT_DIR, "h1_2", "scene.xml"),
)

# Force magnitudes to sweep (Newtons) — extended range
FORCE_MAGNITUDES = [0, 5, 10, 15, 20, 30, 40, 50, 60, 75, 100, 125, 150]

# Directions per magnitude (axis-aligned)
AXIS_DIRECTIONS = {
    "+X": np.array([1, 0, 0], dtype=np.float32),
    "-X": np.array([-1, 0, 0], dtype=np.float32),
    "+Y": np.array([0, 1, 0], dtype=np.float32),
    "-Y": np.array([0, -1, 0], dtype=np.float32),
    "+Z": np.array([0, 0, 1], dtype=np.float32),
    "-Z": np.array([0, 0, -1], dtype=np.float32),
}

# Number of random-direction repeats per (body, magnitude) for statistical robustness
N_RANDOM_REPEATS = 5
RANDOM_SEED = 42

# Bodies to test
FORCE_BODIES = ["torso"]  # Focus on torso for clearest differentiation


# ═══════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrialResult:
    """Comprehensive metrics from one trial."""
    method: str                   # "RMA" or "Baseline"
    body: str                     # which body received force
    force_mag: float              # scalar magnitude (N)
    direction: str                # direction label
    force_vec: List[float]        # [Fx, Fy, Fz]

    survival_time: float          # seconds survived
    success: bool                 # survived full duration

    # Velocity tracking
    tracking_rmse_vx: float
    tracking_rmse_vy: float
    tracking_rmse_xy: float

    # Orientation stability
    mean_orientation_err: float
    max_orientation_err: float

    # Energy & efficiency (post-warmup averages)
    mean_torque_norm: float       # mean ||tau||
    mean_energy: float            # mean sum(|tau_j * dq_j|)
    mean_smoothness: float        # mean ||tau_t - tau_{t-1}||
    mean_jerk: float              # mean ||a_t - 2*a_{t-1} + a_{t-2}||

    # Base height stability
    mean_base_height: float
    std_base_height: float


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def quat_rotate_inverse(q, v):
    """Rotate vector v by inverse of quaternion q (MuJoCo w,x,y,z)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    cw, cx, cy, cz = w, -x, -y, -z
    return np.array([
        v[0]*(cw**2+cx**2-cy**2-cz**2) + v[1]*2*(cx*cy-cw*cz) + v[2]*2*(cx*cz+cw*cy),
        v[0]*2*(cx*cy+cw*cz) + v[1]*(cw**2-cx**2+cy**2-cz**2) + v[2]*2*(cy*cz-cw*cx),
        v[0]*2*(cx*cz-cw*cy) + v[1]*2*(cy*cz+cw*cx) + v[2]*(cw**2-cx**2-cy**2+cz**2),
    ], dtype=np.float32)


def pd_control(target_q, q, kp, dq, kd):
    return (target_q - q) * kp - dq * kd


def compute_obs(d, cfg, action, cmd, phase, n_leg=12):
    quat = d.qpos[3:7].copy()
    omega = d.qvel[3:6].copy()
    qj = d.qpos[7:7+n_leg].copy()
    dqj = d.qvel[6:6+n_leg].copy()

    ang_vel_scaled = omega * cfg["ang_vel_scale"]
    projected_gravity = quat_rotate_inverse(quat, np.array([0., 0., -1.]))

    obs = np.zeros(47, dtype=np.float32)
    obs[0:3] = ang_vel_scaled
    obs[3:6] = projected_gravity
    obs[6:9] = cmd[:3] * cfg["cmd_scale"]
    obs[9:21] = (qj - cfg["default_angles"][:n_leg]) * cfg["dof_pos_scale"]
    obs[21:33] = dqj * cfg["dof_vel_scale"]
    obs[33:45] = action
    obs[45] = np.sin(2 * np.pi * phase)
    obs[46] = np.cos(2 * np.pi * phase)
    return obs, projected_gravity


def _remap_state_dict(model, state_dict):
    model_keys = set(model.state_dict().keys())
    if set(state_dict.keys()) == model_keys:
        return state_dict
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ("actor.", "critic."):
            lp = prefix.replace(".", ".layers.")
            if k.startswith(prefix) and lp not in k and k.replace(prefix, lp, 1) in model_keys:
                new_k = k.replace(prefix, lp, 1)
            elif lp in k and k.replace(lp, prefix, 1) in model_keys:
                new_k = k.replace(lp, prefix, 1)
        remapped[new_k] = v
    return remapped


def load_models(ckpt_path, cfg, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy_cfg = dict(
        num_actor_obs=47 + cfg["rma_latent_dim"],
        num_critic_obs=50 + cfg["rma_latent_dim"],
        num_actions=cfg["num_actions"],
        actor_hidden_dims=[32], critic_hidden_dims=[32],
        rnn_type="lstm", rnn_hidden_size=64, rnn_num_layers=1,
        activation="elu",
    )
    policy = ActorCriticRecurrent(**policy_cfg)
    policy.load_state_dict(_remap_state_dict(policy, ckpt["model_state_dict"]))

    enc_cfg = EnvFactorEncoderCfg(
        in_dim=cfg["rma_et_dim"],
        latent_dim=cfg["rma_latent_dim"],
        hidden_dims=(256, 128),
    )
    encoder = EnvFactorEncoder(enc_cfg)
    encoder.load_state_dict(ckpt["encoder_state_dict"])

    policy.eval()
    encoder.eval()
    print(f"Loaded checkpoint: {ckpt_path} (iter {ckpt.get('iter', '?')})")
    return policy, encoder


# ═══════════════════════════════════════════════════════════════
#  Single trial runner — rich metrics
# ═══════════════════════════════════════════════════════════════

def run_trial(m, policy, encoder, cfg, force_vec, body,
              use_encoder, collect_timeseries=False):
    """Run one headless MuJoCo trial with comprehensive metric collection."""

    dt = cfg["simulation_dt"]
    decim = cfg["control_decimation"]
    n_leg = cfg["num_actions"]
    n_steps = int(cfg["eval_duration"] / dt)
    fall_height = cfg["fall_height"]
    force_start = cfg["force_start_time"]
    warmup = cfg["tracking_warmup"]
    cmd = cfg["cmd"]
    phase_period = cfg["phase_period"]
    latent_dim = cfg["rma_latent_dim"]
    max_tau = cfg["max_tau"]

    d = mujoco.MjData(m)
    n_joints = d.qpos.shape[0] - 7

    # Body IDs
    body_ids = {
        "torso": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link"),
        "left_wrist": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link"),
        "right_wrist": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link"),
    }
    target_body_id = body_ids.get(body, body_ids["torso"])

    default_arms = cfg["default_angles_arms"]
    kps_arm = cfg["kps_arms"]
    kds_arm = cfg["kds_arms"]

    # Reset LSTM
    policy.memory_a.hidden_states = None

    action = np.zeros(n_leg, dtype=np.float32)
    prev_action = np.zeros(n_leg, dtype=np.float32)
    prev_prev_action = np.zeros(n_leg, dtype=np.float32)
    counter = 0
    survival_time = cfg["eval_duration"]

    # Metric accumulators
    vx_errors, vy_errors = [], []
    orientation_errors = []
    orientation_max = 0.0
    torque_norms = []
    energy_vals = []
    smoothness_vals = []
    jerk_vals = []
    base_heights = []
    prev_tau = None

    # Time-series data (optional, for adaptation analysis)
    ts_data = [] if collect_timeseries else None

    for step in range(n_steps):
        t = step * dt

        # Apply forces
        d.xfrc_applied[:] = 0
        if t >= force_start and target_body_id >= 0:
            d.xfrc_applied[target_body_id, :3] = force_vec

        # PD control — legs
        target_dof = action * cfg["action_scale"] + cfg["default_angles"][:n_leg]
        leg_tau = pd_control(target_dof, d.qpos[7:7+n_leg], cfg["kps"],
                             d.qvel[6:6+n_leg], cfg["kds"])
        leg_tau = np.clip(np.nan_to_num(leg_tau), -max_tau, max_tau)
        d.ctrl[:n_leg] = leg_tau

        # PD control — upper body
        if n_joints > n_leg and d.ctrl.shape[0] > n_leg:
            n_upper = min(n_joints - n_leg, d.ctrl.shape[0] - n_leg)
            arm_tau = pd_control(default_arms[:n_upper],
                                 d.qpos[7+n_leg:7+n_leg+n_upper],
                                 kps_arm[:n_upper],
                                 d.qvel[6+n_leg:6+n_leg+n_upper],
                                 kds_arm[:n_upper])
            arm_tau = np.clip(np.nan_to_num(arm_tau), -max_tau, max_tau)
            d.ctrl[n_leg:n_leg+n_upper] = arm_tau

        mujoco.mj_step(m, d)
        counter += 1

        # Check fall
        base_height = d.qpos[2]
        if base_height < fall_height:
            survival_time = t
            break

        # Policy inference at decimation rate
        if counter % decim == 0:
            phase = (t / phase_period) % 1.0
            obs_47, proj_grav = compute_obs(d, cfg, action, cmd, phase, n_leg)

            # Encode forces
            if use_encoder and t >= force_start:
                # Build 9D e_t: only the target body gets force
                e_t = np.zeros(9, dtype=np.float32)
                if body == "torso":
                    e_t[0:3] = force_vec
                elif body == "left_wrist":
                    e_t[3:6] = force_vec
                elif body == "right_wrist":
                    e_t[6:9] = force_vec
            else:
                e_t = np.zeros(9, dtype=np.float32)

            e_t_norm = normalize_et_np(e_t)
            with torch.no_grad():
                z_t = encoder(torch.from_numpy(e_t_norm).unsqueeze(0).float()).numpy().squeeze()

            actor_obs = np.concatenate([obs_47, z_t]).astype(np.float32)
            with torch.no_grad():
                prev_prev_action = prev_action.copy()
                prev_action = action.copy()
                action = policy.act_inference(
                    torch.from_numpy(actor_obs).unsqueeze(0).float()
                ).numpy().squeeze()

            # Collect metrics (after warmup)
            if t >= warmup:
                quat = d.qpos[3:7].copy()
                base_vel_world = d.qvel[0:3].copy()
                base_vel_body = quat_rotate_inverse(quat, base_vel_world)
                vx_errors.append((cmd[0] - base_vel_body[0]) ** 2)
                vy_errors.append((cmd[1] - base_vel_body[1]) ** 2)

                orient_err = np.linalg.norm(proj_grav[:2])
                orientation_errors.append(orient_err)
                orientation_max = max(orientation_max, orient_err)

                base_heights.append(base_height)

                # Torque & energy
                tau_norm = np.linalg.norm(leg_tau)
                torque_norms.append(tau_norm)

                dqj = d.qvel[6:6+n_leg].copy()
                energy = np.sum(np.abs(leg_tau) * np.abs(dqj))
                energy_vals.append(energy)

                # Smoothness (torque derivative)
                if prev_tau is not None:
                    smoothness_vals.append(np.linalg.norm(leg_tau - prev_tau))
                prev_tau = leg_tau.copy()

                # Jerk (action second derivative)
                jerk = np.linalg.norm(action - 2 * prev_action + prev_prev_action)
                jerk_vals.append(jerk)

            # Time-series
            if ts_data is not None and counter % decim == 0:
                ts_data.append({
                    "t": round(t, 4),
                    "base_height": round(float(base_height), 4),
                    "z_t": z_t.tolist(),
                    "orient_err": round(float(np.linalg.norm(proj_grav[:2])), 4),
                    "vx": round(float(quat_rotate_inverse(d.qpos[3:7], d.qvel[0:3])[0]), 4)
                    if t >= warmup else None,
                })

    success = survival_time >= cfg["eval_duration"] - 0.01

    def safe_mean(arr):
        return float(np.mean(arr)) if arr else float('nan')

    def safe_std(arr):
        return float(np.std(arr)) if arr else float('nan')

    return TrialResult(
        method="RMA" if use_encoder else "Baseline",
        body=body,
        force_mag=round(float(np.linalg.norm(force_vec)), 2),
        direction="",
        force_vec=[round(float(x), 3) for x in force_vec],
        survival_time=round(survival_time, 3),
        success=success,
        tracking_rmse_vx=round(np.sqrt(safe_mean(vx_errors)), 4),
        tracking_rmse_vy=round(np.sqrt(safe_mean(vy_errors)), 4),
        tracking_rmse_xy=round(np.sqrt(safe_mean(
            [a + b for a, b in zip(vx_errors, vy_errors)])) if vx_errors else float('nan'), 4),
        mean_orientation_err=round(safe_mean(orientation_errors), 4),
        max_orientation_err=round(orientation_max, 4),
        mean_torque_norm=round(safe_mean(torque_norms), 2),
        mean_energy=round(safe_mean(energy_vals), 2),
        mean_smoothness=round(safe_mean(smoothness_vals), 4),
        mean_jerk=round(safe_mean(jerk_vals), 4),
        mean_base_height=round(safe_mean(base_heights), 4),
        std_base_height=round(safe_std(base_heights), 4),
    ), ts_data


# ═══════════════════════════════════════════════════════════════
#  Trial generation
# ═══════════════════════════════════════════════════════════════

def generate_trials():
    """Generate all trials: RMA + Baseline, axis + random directions."""
    trials = []
    rng = np.random.default_rng(RANDOM_SEED)

    for body in FORCE_BODIES:
        for mag in FORCE_MAGNITUDES:
            if mag == 0:
                # Zero force — one trial per method
                for use_enc in [True, False]:
                    trials.append((
                        np.zeros(3, dtype=np.float32),
                        body, use_enc, f"zero", mag
                    ))
                continue

            # Axis-aligned directions
            for dname, dvec in AXIS_DIRECTIONS.items():
                force = dvec * mag
                for use_enc in [True, False]:
                    trials.append((force.copy(), body, use_enc, dname, mag))

            # Random spherical directions for statistical robustness
            for i in range(N_RANDOM_REPEATS):
                d = rng.standard_normal(3).astype(np.float32)
                d /= np.linalg.norm(d)
                force = d * mag
                for use_enc in [True, False]:
                    trials.append((force.copy(), body, use_enc, f"rnd{i}", mag))

    return trials


# ═══════════════════════════════════════════════════════════════
#  CSV I/O
# ═══════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "method", "body", "force_mag", "direction", "force_fx", "force_fy", "force_fz",
    "survival_time", "success",
    "tracking_rmse_vx", "tracking_rmse_vy", "tracking_rmse_xy",
    "mean_orientation_err", "max_orientation_err",
    "mean_torque_norm", "mean_energy", "mean_smoothness", "mean_jerk",
    "mean_base_height", "std_base_height",
]


def result_to_row(r: TrialResult) -> dict:
    return {
        "method": r.method,
        "body": r.body,
        "force_mag": r.force_mag,
        "direction": r.direction,
        "force_fx": r.force_vec[0],
        "force_fy": r.force_vec[1],
        "force_fz": r.force_vec[2],
        "survival_time": r.survival_time,
        "success": r.success,
        "tracking_rmse_vx": r.tracking_rmse_vx,
        "tracking_rmse_vy": r.tracking_rmse_vy,
        "tracking_rmse_xy": r.tracking_rmse_xy,
        "mean_orientation_err": r.mean_orientation_err,
        "max_orientation_err": r.max_orientation_err,
        "mean_torque_norm": r.mean_torque_norm,
        "mean_energy": r.mean_energy,
        "mean_smoothness": r.mean_smoothness,
        "mean_jerk": r.mean_jerk,
        "mean_base_height": r.mean_base_height,
        "std_base_height": r.std_base_height,
    }


def write_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(result_to_row(r))


def read_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


# ═══════════════════════════════════════════════════════════════
#  Aggregation helpers
# ═══════════════════════════════════════════════════════════════

def aggregate_by_method_and_mag(results):
    """Group results by (method, force_mag) → aggregated stats."""
    groups = {}
    for r in results:
        key = (r.method, r.force_mag)
        groups.setdefault(key, []).append(r)

    agg = {}
    for (method, mag), rs in sorted(groups.items()):
        n = len(rs)
        n_success = sum(1 for r in rs if r.success)
        survivors = [r for r in rs if r.success]

        agg[(method, mag)] = {
            "n_trials": n,
            "success_rate": round(100 * n_success / n, 1),
            "mean_survival": round(np.mean([r.survival_time for r in rs]), 2),
            "std_survival": round(np.std([r.survival_time for r in rs]), 2),
            "mean_track_xy": round(np.nanmean([r.tracking_rmse_xy for r in survivors]), 4) if survivors else float('nan'),
            "std_track_xy": round(np.nanstd([r.tracking_rmse_xy for r in survivors]), 4) if survivors else float('nan'),
            "mean_orient": round(np.nanmean([r.mean_orientation_err for r in survivors]), 4) if survivors else float('nan'),
            "mean_torque": round(np.nanmean([r.mean_torque_norm for r in survivors]), 2) if survivors else float('nan'),
            "mean_energy": round(np.nanmean([r.mean_energy for r in survivors]), 2) if survivors else float('nan'),
            "mean_smoothness": round(np.nanmean([r.mean_smoothness for r in survivors]), 4) if survivors else float('nan'),
            "mean_jerk": round(np.nanmean([r.mean_jerk for r in survivors]), 4) if survivors else float('nan'),
            "mean_height": round(np.nanmean([r.mean_base_height for r in survivors]), 4) if survivors else float('nan'),
        }
    return agg


# ═══════════════════════════════════════════════════════════════
#  Plotting — publication quality
# ═══════════════════════════════════════════════════════════════

def plot_all(results, out_dir):
    """Generate publication-quality comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    # Style for book chapter
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 2,
        "lines.markersize": 7,
    })

    agg = aggregate_by_method_and_mag(results)

    # Extract unique magnitudes
    mags = sorted(set(mag for (_, mag) in agg.keys()))

    # Prepare data arrays
    def get_series(method, key):
        vals, stds = [], []
        for mag in mags:
            d = agg.get((method, mag))
            if d:
                v = d[key]
                vals.append(v if not np.isnan(v) else 0)
                s = d.get(f"std_{key.replace('mean_', '')}", 0)
                stds.append(s if not np.isnan(s) else 0)
            else:
                vals.append(0)
                stds.append(0)
        return np.array(vals), np.array(stds)

    rma_color = "#2196F3"    # blue
    base_color = "#F44336"   # red

    # ── Figure 1: 2x3 comprehensive comparison ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (0,0) Success Rate
    ax = axes[0, 0]
    rma_sr, _ = get_series("RMA", "success_rate")
    base_sr, _ = get_series("Baseline", "success_rate")
    ax.plot(mags, rma_sr, "-o", color=rma_color, label="RMA (with encoder)")
    ax.plot(mags, base_sr, "--s", color=base_color, label="Baseline (no encoder)")
    ax.set_xlabel("Force Magnitude (N)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("(a) Survival Rate")
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # (0,1) Mean Survival Time
    ax = axes[0, 1]
    rma_sv, rma_sv_s = get_series("RMA", "mean_survival")
    base_sv, base_sv_s = get_series("Baseline", "mean_survival")
    # Replace 0 std with small value for clean errorbars
    ax.errorbar(mags, rma_sv, yerr=rma_sv_s, fmt="-o", color=rma_color,
                capsize=3, label="RMA")
    ax.errorbar(mags, base_sv, yerr=base_sv_s, fmt="--s", color=base_color,
                capsize=3, label="Baseline")
    ax.set_xlabel("Force Magnitude (N)")
    ax.set_ylabel("Mean Survival Time (s)")
    ax.set_title("(b) Survival Time")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # (0,2) Tracking RMSE (survivors only)
    ax = axes[0, 2]
    rma_tr, rma_tr_s = get_series("RMA", "mean_track_xy")
    base_tr, base_tr_s = get_series("Baseline", "mean_track_xy")
    # Mask NaN for non-survivors
    rma_mask = ~np.isnan(rma_tr) & (rma_tr > 0)
    base_mask = ~np.isnan(base_tr) & (base_tr > 0)
    m_arr = np.array(mags)
    if rma_mask.any():
        ax.errorbar(m_arr[rma_mask], rma_tr[rma_mask],
                    yerr=rma_tr_s[rma_mask], fmt="-o", color=rma_color,
                    capsize=3, label="RMA")
    if base_mask.any():
        ax.errorbar(m_arr[base_mask], base_tr[base_mask],
                    yerr=base_tr_s[base_mask], fmt="--s", color=base_color,
                    capsize=3, label="Baseline")
    ax.set_xlabel("Force Magnitude (N)")
    ax.set_ylabel("Tracking RMSE (m/s)")
    ax.set_title("(c) Velocity Tracking Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Orientation Error
    ax = axes[1, 0]
    rma_or, _ = get_series("RMA", "mean_orient")
    base_or, _ = get_series("Baseline", "mean_orient")
    rma_mask2 = ~np.isnan(rma_or) & (rma_or > 0)
    base_mask2 = ~np.isnan(base_or) & (base_or > 0)
    if rma_mask2.any():
        ax.plot(m_arr[rma_mask2], rma_or[rma_mask2], "-o", color=rma_color, label="RMA")
    if base_mask2.any():
        ax.plot(m_arr[base_mask2], base_or[base_mask2], "--s", color=base_color, label="Baseline")
    ax.set_xlabel("Force Magnitude (N)")
    ax.set_ylabel("Mean Orientation Error")
    ax.set_title("(d) Orientation Stability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Energy
    ax = axes[1, 1]
    rma_en, _ = get_series("RMA", "mean_energy")
    base_en, _ = get_series("Baseline", "mean_energy")
    rma_mask3 = ~np.isnan(rma_en) & (rma_en > 0)
    base_mask3 = ~np.isnan(base_en) & (base_en > 0)
    if rma_mask3.any():
        ax.plot(m_arr[rma_mask3], rma_en[rma_mask3], "-o", color=rma_color, label="RMA")
    if base_mask3.any():
        ax.plot(m_arr[base_mask3], base_en[base_mask3], "--s", color=base_color, label="Baseline")
    ax.set_xlabel("Force Magnitude (N)")
    ax.set_ylabel("Mean Energy (W)")
    ax.set_title("(e) Energy Consumption")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,2) Smoothness (torque derivative)
    ax = axes[1, 2]
    rma_sm, _ = get_series("RMA", "mean_smoothness")
    base_sm, _ = get_series("Baseline", "mean_smoothness")
    rma_mask4 = ~np.isnan(rma_sm) & (rma_sm > 0)
    base_mask4 = ~np.isnan(base_sm) & (base_sm > 0)
    if rma_mask4.any():
        ax.plot(m_arr[rma_mask4], rma_sm[rma_mask4], "-o", color=rma_color, label="RMA")
    if base_mask4.any():
        ax.plot(m_arr[base_mask4], base_sm[base_mask4], "--s", color=base_color, label="Baseline")
    ax.set_xlabel("Force Magnitude (N)")
    ax.set_ylabel("Mean Torque Smoothness (Nm)")
    ax.set_title("(f) Motion Smoothness")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("RMA vs Baseline: Robustness Under External Force Perturbation\n"
                 "(Torso forces, H1-2 humanoid, MuJoCo sim-to-sim, 10s trials)",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "rma_vs_baseline_comprehensive.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  Saved: {path.replace('.png', '.pdf')}")
    plt.close()

    # ── Figure 2: Success rate + tracking — cleaner 1x2 for book ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(mags, rma_sr, "-o", color=rma_color, label="RMA (with encoder)", linewidth=2.5)
    ax1.plot(mags, base_sr, "--s", color=base_color, label="Baseline ($z_t = 0$)", linewidth=2.5)
    ax1.fill_between(mags, base_sr, rma_sr, alpha=0.1, color=rma_color)
    ax1.set_xlabel("Force Magnitude (N)")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Survival Rate vs. Perturbation Force")
    ax1.set_ylim(-5, 105)
    ax1.legend(loc="lower left", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    if rma_mask.any():
        ax2.errorbar(m_arr[rma_mask], rma_tr[rma_mask],
                     yerr=rma_tr_s[rma_mask], fmt="-o", color=rma_color,
                     capsize=4, label="RMA", linewidth=2.5)
    if base_mask.any():
        ax2.errorbar(m_arr[base_mask], base_tr[base_mask],
                     yerr=base_tr_s[base_mask], fmt="--s", color=base_color,
                     capsize=4, label="Baseline ($z_t = 0$)", linewidth=2.5)
    ax2.set_xlabel("Force Magnitude (N)")
    ax2.set_ylabel("Tracking RMSE (m/s)")
    ax2.set_title("Velocity Tracking Error (survivors only)")
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "rma_vs_baseline_main.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  LaTeX table generation
# ═══════════════════════════════════════════════════════════════

def generate_latex_table(results, out_dir):
    """Generate a LaTeX-ready comparison table."""
    agg = aggregate_by_method_and_mag(results)
    mags = sorted(set(mag for (_, mag) in agg.keys()))

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\caption{RMA vs.\ Baseline (no encoder) under external torso force perturbation "
                 r"in MuJoCo simulation. Each force magnitude is tested over 11 trials "
                 r"(6 axis-aligned + 5 random directions). Metrics are averaged over surviving trials.}")
    lines.append(r"\label{tab:rma_eval}")
    lines.append(r"\begin{tabularx}{\textwidth}{r|cc|cc|cc|cc}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{Force (N)}} & "
                 r"\multicolumn{2}{c|}{\textbf{Success (\%)}} & "
                 r"\multicolumn{2}{c|}{\textbf{Track. RMSE (m/s)}} & "
                 r"\multicolumn{2}{c|}{\textbf{Orient. Error}} & "
                 r"\multicolumn{2}{c}{\textbf{Energy (W)}} \\")
    lines.append(r"& RMA & Base & RMA & Base & RMA & Base & RMA & Base \\")
    lines.append(r"\midrule")

    for mag in mags:
        rma = agg.get(("RMA", mag), {})
        base = agg.get(("Baseline", mag), {})

        def fmt(val, prec=2):
            if isinstance(val, float) and np.isnan(val):
                return "---"
            return f"{val:.{prec}f}"

        rma_sr = fmt(rma.get("success_rate", float('nan')), 1)
        base_sr = fmt(base.get("success_rate", float('nan')), 1)
        rma_tr = fmt(rma.get("mean_track_xy", float('nan')), 3)
        base_tr = fmt(base.get("mean_track_xy", float('nan')), 3)
        rma_or = fmt(rma.get("mean_orient", float('nan')), 3)
        base_or = fmt(base.get("mean_orient", float('nan')), 3)
        rma_en = fmt(rma.get("mean_energy", float('nan')), 1)
        base_en = fmt(base.get("mean_energy", float('nan')), 1)

        # Bold the winner
        def bold_winner(a, b, lower_better=True):
            try:
                va, vb = float(a), float(b)
                if lower_better:
                    if va < vb:
                        return r"\textbf{" + a + "}", b
                    elif vb < va:
                        return a, r"\textbf{" + b + "}"
                else:
                    if va > vb:
                        return r"\textbf{" + a + "}", b
                    elif vb > va:
                        return a, r"\textbf{" + b + "}"
            except ValueError:
                pass
            return a, b

        rma_sr, base_sr = bold_winner(rma_sr, base_sr, lower_better=False)
        rma_tr, base_tr = bold_winner(rma_tr, base_tr, lower_better=True)
        rma_or, base_or = bold_winner(rma_or, base_or, lower_better=True)
        rma_en, base_en = bold_winner(rma_en, base_en, lower_better=True)

        lines.append(
            f"{int(mag)} & {rma_sr} & {base_sr} & "
            f"{rma_tr} & {base_tr} & "
            f"{rma_or} & {base_or} & "
            f"{rma_en} & {base_en} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    path = os.path.join(out_dir, "rma_eval_table.tex")
    with open(path, "w") as f:
        f.write(tex)
    print(f"  Saved LaTeX table: {path}")
    return tex


# ═══════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════

def print_summary(results):
    agg = aggregate_by_method_and_mag(results)
    mags = sorted(set(mag for (_, mag) in agg.keys()))

    print("\n" + "=" * 80)
    print(" BOOK CHAPTER EVALUATION SUMMARY — RMA vs Baseline")
    print("=" * 80)

    header = f"{'Force(N)':>10s} | {'RMA Surv%':>10s} {'Base Surv%':>10s} | " \
             f"{'RMA Track':>10s} {'Base Track':>10s} | " \
             f"{'RMA Orient':>10s} {'Base Orient':>10s} | " \
             f"{'RMA Energy':>10s} {'Base Energy':>10s}"
    print(header)
    print("-" * len(header))

    for mag in mags:
        rma = agg.get(("RMA", mag), {})
        base = agg.get(("Baseline", mag), {})

        def f(v, p=4):
            return f"{v:.{p}f}" if not np.isnan(v) else "N/A"

        print(f"{mag:10.0f} | "
              f"{f(rma.get('success_rate', float('nan')), 1):>10s} "
              f"{f(base.get('success_rate', float('nan')), 1):>10s} | "
              f"{f(rma.get('mean_track_xy', float('nan'))):>10s} "
              f"{f(base.get('mean_track_xy', float('nan'))):>10s} | "
              f"{f(rma.get('mean_orient', float('nan'))):>10s} "
              f"{f(base.get('mean_orient', float('nan'))):>10s} | "
              f"{f(rma.get('mean_energy', float('nan')), 1):>10s} "
              f"{f(base.get('mean_energy', float('nan')), 1):>10s}")

    # Key findings
    print("\n  KEY FINDINGS:")
    for method in ["RMA", "Baseline"]:
        method_results = [r for r in results if r.method == method]
        survived = [r for r in method_results if r.success]
        failed = [r for r in method_results if not r.success]
        print(f"\n  [{method}]")
        print(f"    Total trials: {len(method_results)}")
        print(f"    Survived: {len(survived)} ({100*len(survived)/len(method_results):.1f}%)")
        if failed:
            fail_mags = [r.force_mag for r in failed]
            print(f"    First failure at: {min(fail_mags):.0f} N")
            print(f"    Mean survival at failure: {np.mean([r.survival_time for r in failed]):.2f}s")
        if survived:
            print(f"    Mean tracking RMSE: {np.nanmean([r.tracking_rmse_xy for r in survived]):.4f} m/s")
            print(f"    Mean energy: {np.nanmean([r.mean_energy for r in survived]):.1f} W")
    print("\n" + "=" * 80)


# ═══════════════════════════════════════════════════════════════
#  Time-series adaptation analysis
# ═══════════════════════════════════════════════════════════════

def run_adaptation_analysis(m, policy, encoder, cfg, out_dir):
    """Run a focused trial showing temporal adaptation dynamics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping adaptation analysis")
        return

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
        "legend.fontsize": 11, "figure.dpi": 150, "savefig.dpi": 300,
        "lines.linewidth": 2,
    })

    force_mag = 50.0  # Strong enough to be interesting
    force_vec = np.array([force_mag, 0, 0], dtype=np.float32)  # +X push

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    for use_enc, label, color, ls in [(True, "RMA", "#2196F3", "-"),
                                       (False, "Baseline", "#F44336", "--")]:
        result, ts = run_trial(m, policy, encoder, cfg, force_vec, "torso",
                               use_enc, collect_timeseries=True)

        if not ts:
            continue

        times = [p["t"] for p in ts]
        heights = [p["base_height"] for p in ts]
        orient_errs = [p["orient_err"] for p in ts]
        vxs = [p["vx"] if p["vx"] is not None else float('nan') for p in ts]

        axes[0].plot(times, heights, ls, color=color, label=label)
        axes[1].plot(times, orient_errs, ls, color=color, label=label)
        axes[2].plot(times, vxs, ls, color=color, label=label)

        # z_t components (only for RMA)
        if use_enc and ts:
            z_ts = np.array([p["z_t"] for p in ts])
            for dim in range(min(4, z_ts.shape[1])):
                axes[3].plot(times, z_ts[:, dim], label=f"$z_{{{dim}}}$", alpha=0.8)

    force_start = cfg["force_start_time"]
    for ax in axes:
        ax.axvline(x=force_start, color="gray", linestyle=":", alpha=0.7, label="Force onset" if ax == axes[0] else "")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Base Height (m)")
    axes[0].set_title("(a) Base Height")
    axes[0].legend(loc="lower left")

    axes[1].set_ylabel("Orientation Error")
    axes[1].set_title("(b) Orientation Stability")
    axes[1].legend(loc="upper left")

    axes[2].set_ylabel("Forward Velocity (m/s)")
    axes[2].axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="Command (0.5 m/s)")
    axes[2].set_title("(c) Velocity Tracking")
    axes[2].legend(loc="lower left")

    axes[3].set_ylabel("Extrinsics $\\hat{z}_t$")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("(d) Adaptation Module Latent (RMA only)")
    axes[3].legend(loc="upper right", ncol=2)

    fig.suptitle(f"Temporal Adaptation Analysis: {force_mag:.0f}N +X Torso Push\n"
                 f"(Force applied at t={force_start}s)",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "adaptation_analysis.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RMA Book Chapter Evaluation")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Training checkpoint path")
    parser.add_argument("--plot_only", action="store_true",
                        help="Re-plot from existing CSV")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(_SCRIPT_DIR, "book_eval_results"))
    args = parser.parse_args()

    cfg = EVAL_CONFIG
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "rma_eval_results.csv")

    if args.plot_only:
        print(f"Re-plotting from {csv_path}")
        rows = read_csv(csv_path)
        # Reconstruct TrialResult objects
        results = []
        for r in rows:
            results.append(TrialResult(
                method=r["method"], body=r["body"],
                force_mag=float(r["force_mag"]), direction=r["direction"],
                force_vec=[float(r["force_fx"]), float(r["force_fy"]), float(r["force_fz"])],
                survival_time=float(r["survival_time"]),
                success=r["success"] == "True",
                tracking_rmse_vx=float(r["tracking_rmse_vx"]),
                tracking_rmse_vy=float(r["tracking_rmse_vy"]),
                tracking_rmse_xy=float(r["tracking_rmse_xy"]),
                mean_orientation_err=float(r["mean_orientation_err"]),
                max_orientation_err=float(r["max_orientation_err"]),
                mean_torque_norm=float(r["mean_torque_norm"]),
                mean_energy=float(r["mean_energy"]),
                mean_smoothness=float(r["mean_smoothness"]),
                mean_jerk=float(r["mean_jerk"]),
                mean_base_height=float(r["mean_base_height"]),
                std_base_height=float(r["std_base_height"]),
            ))
        print_summary(results)
        plot_all(results, out_dir)
        generate_latex_table(results, out_dir)
        return

    # Resolve checkpoint
    ckpt_path = args.ckpt
    if not ckpt_path:
        # Default: latest RMA checkpoint
        log_dir = os.path.join(_REPO_ROOT, "logs", "h1_2_rma", "Apr05_19-19-47_")
        if os.path.isdir(log_dir):
            pts = sorted([f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")],
                         key=lambda x: int(x.replace("model_", "").replace(".pt", "")))
            if pts:
                ckpt_path = os.path.join(log_dir, pts[-1])

    if not ckpt_path or not os.path.isfile(ckpt_path):
        print("ERROR: No checkpoint found. Use --ckpt <path>")
        sys.exit(1)

    # Load models
    policy, encoder = load_models(ckpt_path, cfg)

    # Load MuJoCo model
    m = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    m.opt.timestep = cfg["simulation_dt"]

    # Generate and run trials
    trials = generate_trials()
    n_total = len(trials)

    print(f"\n{'='*60}")
    print(f" EVALUATION: {n_total} trials")
    print(f"   Force magnitudes: {FORCE_MAGNITUDES}")
    print(f"   Directions: 6 axis + {N_RANDOM_REPEATS} random = 11 per mag")
    print(f"   Methods: RMA + Baseline = x2")
    print(f"   Bodies: {FORCE_BODIES}")
    print(f"   Duration: {cfg['eval_duration']}s per trial")
    est_time = n_total * cfg['eval_duration'] * 0.12
    print(f"   Estimated wall time: ~{est_time/60:.1f} min")
    print(f"{'='*60}\n")

    results = []
    t_start = time.time()

    for i, (force_vec, body, use_enc, direction, mag) in enumerate(trials):
        method = "RMA" if use_enc else "Baseline"
        r, _ = run_trial(m, policy, encoder, cfg, force_vec, body, use_enc)
        r.direction = direction
        results.append(r)

        status = "OK" if r.success else f"FALL@{r.survival_time:.1f}s"
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (n_total - i - 1)
        print(f"  [{i+1:4d}/{n_total}] {method:>8s} | {mag:5.0f}N {direction:>5s} | "
              f"{status:>12s} | track={r.tracking_rmse_xy:.4f} | "
              f"energy={r.mean_energy:.1f} | "
              f"[{elapsed:.0f}s, ~{eta:.0f}s left]")

    # Save CSV
    write_csv(results, csv_path)
    print(f"\nResults saved: {csv_path}")

    # Summary
    print_summary(results)

    # Plots
    plot_all(results, out_dir)

    # LaTeX table
    tex = generate_latex_table(results, out_dir)
    print("\nLaTeX table preview:")
    print(tex[:500] + "...")

    # Adaptation analysis (separate run with time-series)
    print("\nRunning temporal adaptation analysis...")
    run_adaptation_analysis(m, policy, encoder, cfg, out_dir)

    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
