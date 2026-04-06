"""RMA Force Sweep — Robustness evaluation across 9D force space.

Runs headless MuJoCo trials sweeping external forces on torso + wrists,
measuring survival rate and velocity tracking under perturbation.

Usage:
  python sweep_rma_forces.py --config sweep_config.yaml
  python sweep_rma_forces.py --config sweep_config.yaml --ckpt ../logs/h1_2_rma/<run>/model_5000.pt
  python sweep_rma_forces.py --config sweep_config.yaml --plot_only   # re-plot from existing CSV
"""

import os
import sys
import argparse
import csv
import time
from dataclasses import dataclass, field
from typing import List, Optional

import yaml
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mujoco

from rsl_rl.modules import ActorCriticRecurrent
from rma.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg


# ──────────────────────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class TrialSpec:
    """Single trial: which forces to apply, with what encoder mode."""
    torso_force: np.ndarray       # (3,)
    left_wrist_force: np.ndarray  # (3,)
    right_wrist_force: np.ndarray # (3,)
    use_encoder: bool = True
    label: str = ""


@dataclass
class TrialResult:
    """Metrics from one trial."""
    label: str
    torso_force: List[float]
    left_wrist_force: List[float]
    right_wrist_force: List[float]
    total_force_mag: float
    use_encoder: bool
    survival_time: float          # seconds survived
    success: bool                 # survived >= eval_duration
    tracking_rmse_vx: float
    tracking_rmse_vy: float
    tracking_rmse_xy: float       # combined XY RMSE
    mean_orientation_err: float   # mean |projected_gravity[:2]|


# ──────────────────────────────────────────────────────────────
#  Helpers (shared with mujoco_deploy_rma.py)
# ──────────────────────────────────────────────────────────────

def quat_rotate_inverse(q, v):
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
            layers_prefix = prefix.replace(".", ".layers.")
            if k.startswith(prefix) and layers_prefix not in k and k.replace(prefix, layers_prefix, 1) in model_keys:
                new_k = k.replace(prefix, layers_prefix, 1)
            elif layers_prefix in k and k.replace(layers_prefix, prefix, 1) in model_keys:
                new_k = k.replace(layers_prefix, prefix, 1)
        remapped[new_k] = v
    return remapped


# ──────────────────────────────────────────────────────────────
#  Model loading
# ──────────────────────────────────────────────────────────────

def load_models(cfg, ckpt_path=None, device="cpu"):
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        policy_cfg = dict(
            num_actor_obs=47 + cfg.get("rma_latent_dim", 8),
            num_critic_obs=50 + cfg.get("rma_latent_dim", 8),
            num_actions=cfg["num_actions"],
            actor_hidden_dims=[32], critic_hidden_dims=[32],
            rnn_type="lstm", rnn_hidden_size=64, rnn_num_layers=1,
            activation="elu",
        )
        policy = ActorCriticRecurrent(**policy_cfg)
        policy.load_state_dict(_remap_state_dict(policy, ckpt["model_state_dict"]))
        enc_cfg = EnvFactorEncoderCfg(
            in_dim=cfg.get("rma_et_dim", 9),
            latent_dim=cfg.get("rma_latent_dim", 8),
            hidden_dims=(256, 128),
        )
        encoder = EnvFactorEncoder(enc_cfg)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
    else:
        pol_data = torch.load(cfg["policy_path"], map_location=device)
        policy = ActorCriticRecurrent(**pol_data["cfg"])
        policy.load_state_dict(_remap_state_dict(policy, pol_data["model_state_dict"]))
        enc_data = torch.load(cfg["encoder_path"], map_location=device)
        encoder = EnvFactorEncoder(EnvFactorEncoderCfg(**enc_data["cfg"]))
        encoder.load_state_dict(enc_data["encoder_state_dict"])

    policy.eval()
    encoder.eval()
    return policy, encoder


# ──────────────────────────────────────────────────────────────
#  Single trial runner
# ──────────────────────────────────────────────────────────────

def run_trial(m_template, policy, encoder, cfg, trial: TrialSpec) -> TrialResult:
    """Run one headless MuJoCo trial, return metrics."""
    eval_cfg = cfg["eval"]
    dt = cfg["simulation_dt"]
    decim = cfg["control_decimation"]
    n_leg = cfg["num_actions"]
    n_steps = int(eval_cfg["duration"] / dt)
    fall_height = eval_cfg["fall_height"]
    force_start = eval_cfg.get("force_start_time", 0.0)
    tracking_warmup = eval_cfg.get("tracking_warmup", 2.0)
    cmd = np.array(eval_cfg["cmd"], dtype=np.float32)
    phase_period = cfg.get("phase_period", 0.8)
    latent_dim = cfg.get("rma_latent_dim", 8)
    max_tau = 300.0

    # Fresh sim data
    d = mujoco.MjData(m_template)
    n_joints = d.qpos.shape[0] - 7

    # Body IDs
    torso_id = mujoco.mj_name2id(m_template, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    left_id = mujoco.mj_name2id(m_template, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
    right_id = mujoco.mj_name2id(m_template, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")

    # Upper-body PD
    default_arms = np.array(cfg.get("default_angles_arms",
                                     np.zeros(n_joints - n_leg)), dtype=np.float32)
    kps_arm = np.array(cfg.get("kps_arms",
                                np.ones(n_joints - n_leg) * 100), dtype=np.float32)
    kds_arm = np.array(cfg.get("kds_arms",
                                np.ones(n_joints - n_leg) * 5), dtype=np.float32)

    # Reset LSTM hidden state
    policy.memory_a.hidden_states = None

    action = np.zeros(n_leg, dtype=np.float32)
    counter = 0
    survival_time = eval_cfg["duration"]

    # Tracking metrics
    vx_errors = []
    vy_errors = []
    orientation_errors = []

    for step in range(n_steps):
        t = step * dt

        # Apply forces
        d.xfrc_applied[:] = 0
        if t >= force_start:
            if torso_id >= 0:
                d.xfrc_applied[torso_id, :3] = trial.torso_force
            if left_id >= 0:
                d.xfrc_applied[left_id, :3] = trial.left_wrist_force
            if right_id >= 0:
                d.xfrc_applied[right_id, :3] = trial.right_wrist_force

        # PD legs
        target_dof = action * cfg["action_scale"] + np.array(cfg["default_angles"][:n_leg], dtype=np.float32)
        leg_tau = pd_control(target_dof, d.qpos[7:7+n_leg], np.array(cfg["kps"], dtype=np.float32),
                             d.qvel[6:6+n_leg], np.array(cfg["kds"], dtype=np.float32))
        leg_tau = np.clip(np.nan_to_num(leg_tau), -max_tau, max_tau)
        d.ctrl[:n_leg] = leg_tau

        # PD upper body
        if n_joints > n_leg and d.ctrl.shape[0] > n_leg:
            n_upper = min(n_joints - n_leg, d.ctrl.shape[0] - n_leg)
            arm_tau = pd_control(default_arms[:n_upper], d.qpos[7+n_leg:7+n_leg+n_upper],
                                 kps_arm[:n_upper], d.qvel[6+n_leg:6+n_leg+n_upper], kds_arm[:n_upper])
            arm_tau = np.clip(np.nan_to_num(arm_tau), -max_tau, max_tau)
            d.ctrl[n_leg:n_leg+n_upper] = arm_tau

        mujoco.mj_step(m_template, d)
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

            # Encode
            if trial.use_encoder and t >= force_start:
                e_t = np.concatenate([trial.torso_force, trial.left_wrist_force,
                                      trial.right_wrist_force]).astype(np.float32)
            else:
                e_t = np.zeros(cfg.get("rma_et_dim", 9), dtype=np.float32)

            with torch.no_grad():
                z_t = encoder(torch.from_numpy(e_t).unsqueeze(0).float()).numpy().squeeze()

            actor_obs = np.concatenate([obs_47, z_t]).astype(np.float32)
            with torch.no_grad():
                action = policy.act_inference(
                    torch.from_numpy(actor_obs).unsqueeze(0).float()
                ).numpy().squeeze()

            # Collect tracking metrics (after warmup)
            if t >= tracking_warmup:
                quat = d.qpos[3:7].copy()
                # World-frame base velocity
                base_vel_world = d.qvel[0:3].copy()
                # Rotate to body frame for comparison with commands
                base_vel_body = quat_rotate_inverse(quat, base_vel_world)
                vx_errors.append((cmd[0] - base_vel_body[0]) ** 2)
                vy_errors.append((cmd[1] - base_vel_body[1]) ** 2)
                orientation_errors.append(np.linalg.norm(proj_grav[:2]))

    success = survival_time >= eval_cfg["duration"] - 0.01
    rmse_vx = np.sqrt(np.mean(vx_errors)) if vx_errors else float('nan')
    rmse_vy = np.sqrt(np.mean(vy_errors)) if vy_errors else float('nan')
    rmse_xy = np.sqrt(np.mean(np.array(vx_errors) + np.array(vy_errors))) if vx_errors else float('nan')
    mean_orient = np.mean(orientation_errors) if orientation_errors else float('nan')

    total_mag = (np.linalg.norm(trial.torso_force) +
                 np.linalg.norm(trial.left_wrist_force) +
                 np.linalg.norm(trial.right_wrist_force))

    return TrialResult(
        label=trial.label,
        torso_force=trial.torso_force.tolist(),
        left_wrist_force=trial.left_wrist_force.tolist(),
        right_wrist_force=trial.right_wrist_force.tolist(),
        total_force_mag=total_mag,
        use_encoder=trial.use_encoder,
        survival_time=round(survival_time, 3),
        success=success,
        tracking_rmse_vx=round(rmse_vx, 4),
        tracking_rmse_vy=round(rmse_vy, 4),
        tracking_rmse_xy=round(rmse_xy, 4),
        mean_orientation_err=round(mean_orient, 4),
    )


# ──────────────────────────────────────────────────────────────
#  Sweep trial generation
# ──────────────────────────────────────────────────────────────

AXIS_DIRECTIONS = {
    "+X": np.array([1, 0, 0], dtype=np.float32),
    "-X": np.array([-1, 0, 0], dtype=np.float32),
    "+Y": np.array([0, 1, 0], dtype=np.float32),
    "-Y": np.array([0, -1, 0], dtype=np.float32),
    "+Z": np.array([0, 0, 1], dtype=np.float32),
    "-Z": np.array([0, 0, -1], dtype=np.float32),
}

BODY_KEY_MAP = {
    "torso": "torso_force",
    "left_wrist": "left_wrist_force",
    "right_wrist": "right_wrist_force",
}


def sample_sphere(n, rng):
    """Sample n unit vectors uniformly on the sphere."""
    d = rng.standard_normal((n, 3)).astype(np.float32)
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-6, None)
    return d / norms


def generate_single_body_trials(cfg) -> List[TrialSpec]:
    """Generate trials sweeping one body at a time."""
    trials = []
    magnitudes = cfg["magnitudes"]
    mode = cfg.get("sweep_mode", "axis")
    sph_n = cfg.get("spherical_n_samples", 10)
    sph_seed = cfg.get("spherical_seed", 42)
    rng = np.random.default_rng(sph_seed)
    zero = np.zeros(3, dtype=np.float32)

    for body in cfg["sweep_bodies"]:
        bname = body["name"]
        blabel = body["label"]

        for mag in magnitudes:
            if mag == 0:
                # Zero force: one trial
                trials.append(TrialSpec(
                    torso_force=zero.copy(), left_wrist_force=zero.copy(),
                    right_wrist_force=zero.copy(), use_encoder=True,
                    label=f"{blabel}|0N|zero",
                ))
                continue

            directions = {}
            if mode in ("axis", "both"):
                directions.update(AXIS_DIRECTIONS)
            if mode in ("spherical", "both"):
                sphere_dirs = sample_sphere(sph_n, rng)
                for i, d in enumerate(sphere_dirs):
                    directions[f"sph{i}"] = d

            for dname, dvec in directions.items():
                force = dvec * mag
                kwargs = {k: zero.copy() for k in BODY_KEY_MAP.values()}
                kwargs[BODY_KEY_MAP[bname]] = force
                trials.append(TrialSpec(
                    **kwargs, use_encoder=True,
                    label=f"{blabel}|{mag}N|{dname}",
                ))

    return trials


def generate_combined_trials(cfg) -> List[TrialSpec]:
    """Generate trials with forces on all 3 bodies simultaneously."""
    trials = []
    combos = cfg.get("combined_sweeps", [])
    n_samp = cfg.get("combined_n_samples", 5)
    seed = cfg.get("combined_seed", 123)
    rng = np.random.default_rng(seed)

    for combo in combos:
        torso_mag, left_mag, right_mag = combo
        for i in range(n_samp):
            dirs = sample_sphere(3, rng)
            trials.append(TrialSpec(
                torso_force=(dirs[0] * torso_mag).astype(np.float32),
                left_wrist_force=(dirs[1] * left_mag).astype(np.float32),
                right_wrist_force=(dirs[2] * right_mag).astype(np.float32),
                use_encoder=True,
                label=f"combined|{torso_mag}/{left_mag}/{right_mag}N|s{i}",
            ))

    return trials


def generate_baseline_trials(cfg) -> List[TrialSpec]:
    """Generate no-encode baseline trials for comparison."""
    if not cfg.get("baseline_no_encode", False):
        return []

    trials = []
    mags = cfg.get("baseline_magnitudes", [0, 5, 10, 15])
    zero = np.zeros(3, dtype=np.float32)
    rng = np.random.default_rng(99)

    for body in cfg["sweep_bodies"]:
        bname = body["name"]
        blabel = body["label"]
        for mag in mags:
            if mag == 0:
                trials.append(TrialSpec(
                    torso_force=zero.copy(), left_wrist_force=zero.copy(),
                    right_wrist_force=zero.copy(), use_encoder=False,
                    label=f"NO_ENC|{blabel}|0N|zero",
                ))
                continue
            # Test along +X and -Z (representative)
            for dname, dvec in [("+X", np.array([1,0,0], dtype=np.float32)),
                                ("-Z", np.array([0,0,-1], dtype=np.float32))]:
                force = dvec * mag
                kwargs = {k: zero.copy() for k in BODY_KEY_MAP.values()}
                kwargs[BODY_KEY_MAP[bname]] = force
                trials.append(TrialSpec(
                    **kwargs, use_encoder=False,
                    label=f"NO_ENC|{blabel}|{mag}N|{dname}",
                ))

    return trials


# ──────────────────────────────────────────────────────────────
#  CSV I/O
# ──────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "label", "use_encoder",
    "torso_fx", "torso_fy", "torso_fz",
    "left_fx", "left_fy", "left_fz",
    "right_fx", "right_fy", "right_fz",
    "total_force_mag",
    "survival_time", "success",
    "tracking_rmse_vx", "tracking_rmse_vy", "tracking_rmse_xy",
    "mean_orientation_err",
]


def result_to_row(r: TrialResult) -> dict:
    return {
        "label": r.label,
        "use_encoder": r.use_encoder,
        "torso_fx": r.torso_force[0], "torso_fy": r.torso_force[1], "torso_fz": r.torso_force[2],
        "left_fx": r.left_wrist_force[0], "left_fy": r.left_wrist_force[1], "left_fz": r.left_wrist_force[2],
        "right_fx": r.right_wrist_force[0], "right_fy": r.right_wrist_force[1], "right_fz": r.right_wrist_force[2],
        "total_force_mag": r.total_force_mag,
        "survival_time": r.survival_time,
        "success": r.success,
        "tracking_rmse_vx": r.tracking_rmse_vx,
        "tracking_rmse_vy": r.tracking_rmse_vy,
        "tracking_rmse_xy": r.tracking_rmse_xy,
        "mean_orientation_err": r.mean_orientation_err,
    }


def write_csv(results: List[TrialResult], path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(result_to_row(r))


def read_csv(path: str) -> List[dict]:
    with open(path, "r") as f:
        return list(csv.DictReader(f))


# ──────────────────────────────────────────────────────────────
#  Summary + plotting
# ──────────────────────────────────────────────────────────────

def print_summary(results: List[TrialResult]):
    enc_results = [r for r in results if r.use_encoder]
    noenc_results = [r for r in results if not r.use_encoder]

    print("\n" + "=" * 70)
    print(" SWEEP SUMMARY")
    print("=" * 70)

    for tag, subset in [("RMA (encoder)", enc_results), ("No encoder (baseline)", noenc_results)]:
        if not subset:
            continue
        n_total = len(subset)
        n_success = sum(1 for r in subset if r.success)
        rate = n_success / n_total * 100
        surv = [r.survival_time for r in subset]
        track = [r.tracking_rmse_xy for r in subset if r.success and not np.isnan(r.tracking_rmse_xy)]
        orient = [r.mean_orientation_err for r in subset if r.success and not np.isnan(r.mean_orientation_err)]

        print(f"\n  [{tag}]  {n_success}/{n_total} survived ({rate:.1f}%)")
        print(f"    Survival time: mean={np.mean(surv):.2f}s, min={np.min(surv):.2f}s")
        if track:
            print(f"    Tracking RMSE (survivors): mean={np.mean(track):.4f}, max={np.max(track):.4f} m/s")
        if orient:
            print(f"    Orientation err (survivors): mean={np.mean(orient):.4f}")

    # Per-body breakdown
    print("\n  Per-body success rate (RMA encoder, axis+spherical):")
    for r in results:
        if not r.use_encoder:
            continue
    bodies = set()
    for r in enc_results:
        parts = r.label.split("|")
        if len(parts) >= 1 and parts[0] not in ("combined",):
            bodies.add(parts[0])

    for body in sorted(bodies):
        body_results = [r for r in enc_results if r.label.startswith(body + "|")]
        if not body_results:
            continue
        # Group by magnitude
        mag_groups = {}
        for r in body_results:
            parts = r.label.split("|")
            mag_str = parts[1] if len(parts) > 1 else "?"
            mag_groups.setdefault(mag_str, []).append(r)

        print(f"\n    {body}:")
        for mag_str in sorted(mag_groups.keys(), key=lambda s: float(s.replace("N", "")) if s.replace("N", "").replace(".", "").isdigit() else 0):
            group = mag_groups[mag_str]
            n_s = sum(1 for r in group if r.success)
            n_t = len(group)
            track_vals = [r.tracking_rmse_xy for r in group if r.success and not np.isnan(r.tracking_rmse_xy)]
            track_str = f", track_rmse={np.mean(track_vals):.4f}" if track_vals else ""
            print(f"      {mag_str:>6s}: {n_s}/{n_t} ({100*n_s/n_t:5.1f}%){track_str}")

    print("\n" + "=" * 70)


def plot_results(csv_path: str, out_dir: str):
    """Generate plots from sweep CSV."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    rows = read_csv(csv_path)
    if not rows:
        return

    # --- Plot 1: Success rate vs magnitude (per body, encoder vs no-encoder) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    body_names = ["Torso", "Left Wrist", "Right Wrist"]

    for ax, bname in zip(axes, body_names):
        for enc_val, style, clr, lbl in [("True", "-o", "C0", "RMA"), ("False", "--s", "C3", "No encoder")]:
            body_rows = [r for r in rows if r["label"].startswith(bname) or
                         r["label"].startswith(f"NO_ENC|{bname}")]
            body_rows = [r for r in body_rows if str(r["use_encoder"]) == enc_val]

            mag_success = {}
            for r in body_rows:
                parts = r["label"].split("|")
                mag_str = parts[1] if "NO_ENC" not in parts[0] else parts[2]
                mag = float(mag_str.replace("N", ""))
                mag_success.setdefault(mag, []).append(r["success"] == "True")

            if not mag_success:
                continue
            mags_sorted = sorted(mag_success.keys())
            rates = [100 * sum(mag_success[m]) / len(mag_success[m]) for m in mags_sorted]
            ax.plot(mags_sorted, rates, style, color=clr, label=lbl, markersize=6)

        ax.set_xlabel("Force magnitude (N)")
        ax.set_title(bname)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Success rate (%)")

    fig.suptitle("Survival rate vs force magnitude (10s trials)")
    plt.tight_layout()
    path = os.path.join(out_dir, "success_rate_vs_magnitude.png")
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()

    # --- Plot 2: Tracking RMSE vs magnitude (survivors only) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, bname in zip(axes, body_names):
        enc_rows = [r for r in rows if r["label"].startswith(bname) and r["use_encoder"] == "True"
                     and r["success"] == "True"]
        mag_track = {}
        for r in enc_rows:
            mag_str = r["label"].split("|")[1]
            mag = float(mag_str.replace("N", ""))
            val = float(r["tracking_rmse_xy"])
            if not np.isnan(val):
                mag_track.setdefault(mag, []).append(val)

        if mag_track:
            mags_sorted = sorted(mag_track.keys())
            means = [np.mean(mag_track[m]) for m in mags_sorted]
            stds = [np.std(mag_track[m]) for m in mags_sorted]
            ax.errorbar(mags_sorted, means, yerr=stds, fmt="-o", color="C0",
                        capsize=3, markersize=6)

        ax.set_xlabel("Force magnitude (N)")
        ax.set_title(bname)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Tracking RMSE (m/s)")

    fig.suptitle("Velocity tracking error vs force (survivors only)")
    plt.tight_layout()
    path = os.path.join(out_dir, "tracking_rmse_vs_magnitude.png")
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()

    # --- Plot 3: Combined sweep results ---
    combined_rows = [r for r in rows if r["label"].startswith("combined")]
    if combined_rows:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        total_mags = [float(r["total_force_mag"]) for r in combined_rows]
        survivals = [float(r["survival_time"]) for r in combined_rows]
        successes = [r["success"] == "True" for r in combined_rows]
        colors = ["C0" if s else "C3" for s in successes]

        ax1.scatter(total_mags, survivals, c=colors, alpha=0.7, edgecolors="k", linewidth=0.5)
        ax1.axhline(y=float(rows[0].get("survival_time", 10)), color="gray", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Total force magnitude (N)")
        ax1.set_ylabel("Survival time (s)")
        ax1.set_title("Combined forces: survival")
        ax1.grid(True, alpha=0.3)

        track_vals = [float(r["tracking_rmse_xy"]) for r in combined_rows
                      if r["success"] == "True" and r["tracking_rmse_xy"] != "nan"]
        track_mags = [float(r["total_force_mag"]) for r in combined_rows
                      if r["success"] == "True" and r["tracking_rmse_xy"] != "nan"]
        if track_vals:
            ax2.scatter(track_mags, track_vals, c="C0", alpha=0.7, edgecolors="k", linewidth=0.5)
        ax2.set_xlabel("Total force magnitude (N)")
        ax2.set_ylabel("Tracking RMSE (m/s)")
        ax2.set_title("Combined forces: tracking (survivors)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(out_dir, "combined_sweep.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
        plt.close()


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RMA force sweep evaluation")
    parser.add_argument("--config", type=str, default=os.path.join(_SCRIPT_DIR, "sweep_config.yaml"))
    parser.add_argument("--ckpt", type=str, default=None, help="Direct checkpoint path")
    parser.add_argument("--plot_only", action="store_true", help="Re-plot from existing CSV")
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(_SCRIPT_DIR, args.config)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    config_dir = os.path.dirname(os.path.abspath(config_path))
    for key in ("policy_path", "encoder_path", "xml_path"):
        if key in cfg and cfg[key] and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.normpath(os.path.join(config_dir, cfg[key]))

    # Convert array configs
    for key in ("kps", "kds", "kps_arms", "kds_arms", "default_angles",
                "default_angles_arms", "cmd_scale"):
        if key in cfg:
            cfg[key] = np.array(cfg[key], dtype=np.float32)

    out_dir = os.path.join(config_dir, cfg.get("output_dir", "sweep_results"))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "sweep_results.csv")

    if args.plot_only:
        print(f"Re-plotting from {csv_path}")
        plot_results(csv_path, out_dir)
        return

    # Load models
    ckpt_path = args.ckpt
    if ckpt_path and not os.path.isabs(ckpt_path):
        ckpt_path = os.path.normpath(os.path.join(config_dir, ckpt_path))
    if not ckpt_path and "ckpt_path" in cfg and cfg["ckpt_path"]:
        ckpt_path = cfg["ckpt_path"]
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.normpath(os.path.join(config_dir, ckpt_path))

    policy, encoder = load_models(cfg, ckpt_path)

    # Load MuJoCo model
    m = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    m.opt.timestep = cfg["simulation_dt"]

    # Generate trials
    trials = []
    trials += generate_single_body_trials(cfg)
    trials += generate_combined_trials(cfg)
    trials += generate_baseline_trials(cfg)

    # De-duplicate zero-force trials
    seen_labels = set()
    unique_trials = []
    for t in trials:
        if t.label in seen_labels:
            continue
        seen_labels.add(t.label)
        unique_trials.append(t)
    trials = unique_trials

    print(f"\nTotal trials: {len(trials)}")
    print(f"  Single-body axis/spherical: {sum(1 for t in trials if t.use_encoder and 'combined' not in t.label)}")
    print(f"  Combined: {sum(1 for t in trials if 'combined' in t.label)}")
    print(f"  No-encoder baseline: {sum(1 for t in trials if not t.use_encoder)}")
    print(f"  Eval duration: {cfg['eval']['duration']}s per trial")
    est_time = len(trials) * cfg['eval']['duration'] * 0.15  # rough: sim ~6x faster than real-time
    print(f"  Estimated wall time: ~{est_time/60:.1f} min\n")

    # Run all trials
    results = []
    t_start = time.time()
    for i, trial in enumerate(trials):
        r = run_trial(m, policy, encoder, cfg, trial)
        results.append(r)
        status = "OK" if r.success else f"FALL@{r.survival_time:.1f}s"
        enc_tag = "RMA" if r.use_encoder else "NO_ENC"
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(trials) - i - 1)
        print(f"  [{i+1:4d}/{len(trials)}] {status:>10s} | {enc_tag:>6s} | "
              f"track={r.tracking_rmse_xy:.4f} | {r.label}  "
              f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    # Save results
    write_csv(results, csv_path)
    print(f"\nResults saved: {csv_path}")

    # Summary
    print_summary(results)

    # Plots
    plot_results(csv_path, out_dir)

    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
