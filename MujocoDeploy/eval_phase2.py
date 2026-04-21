"""Phase 2 (RMA adaptation) evaluation — head-to-head vs teacher and baseline.

For each force perturbation, runs three modes back-to-back on identical seeds:

  1. teacher    : z_t = encoder(true e_t)        — Phase 1, privileged info
  2. adaptation : z_t = adaptation_module(hist)  — Phase 2, deployable
  3. baseline   : z_t = 0                        — no env-factor info at all

Outputs a CSV with one row per (trial, mode) and prints a summary
comparing per-mode success / tracking RMSE / orientation error.

Usage:
  python eval_phase2.py \
      --policy_ckpt ../logs/h1_2_rma/Apr19_14-17-35_paperfull_v1/model_6000.pt \
      --adaptation_ckpt ../logs/h1_2_rma_phase2/Apr19_21-41-31_phase2_v1/adaptation_5000.pt \
      --config sweep_config_m6000.yaml \
      --device cuda
"""

import os
import sys
import argparse
import csv
import time
from dataclasses import dataclass
from typing import List

import yaml
import numpy as np
import torch
import mujoco

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rsl_rl.modules import ActorCriticRecurrent
from rma.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg
from rma.adaptation_module import Adaptation1DCNN, Adaptation1DCNNCfg

# Reuse sweep helpers
from sweep_rma_forces import (
    quat_rotate_inverse, pd_control, compute_obs,
    _remap_state_dict, normalize_et_np,
    AXIS_DIRECTIONS, BODY_KEY_MAP, sample_sphere,
)


MODES = ("teacher", "adaptation", "baseline")


@dataclass
class EvalTrial:
    torso_force: np.ndarray
    left_wrist_force: np.ndarray
    right_wrist_force: np.ndarray
    label: str


@dataclass
class EvalResult:
    label: str
    mode: str
    torso_force: List[float]
    left_wrist_force: List[float]
    right_wrist_force: List[float]
    total_force_mag: float
    survival_time: float
    success: bool
    tracking_rmse_xy: float
    mean_orientation_err: float
    mean_z_l2_to_teacher: float   # only meaningful for adaptation/baseline


# ──────────────────────────────────────────────────────────────
#  Model loading
# ──────────────────────────────────────────────────────────────

def load_phase1(policy_ckpt: str, cfg: dict, device: str):
    ckpt = torch.load(policy_ckpt, map_location=device)
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
        in_dim=cfg.get("rma_et_dim", 26),
        latent_dim=cfg.get("rma_latent_dim", 8),
        hidden_dims=(256, 128),
    )
    encoder = EnvFactorEncoder(enc_cfg)
    encoder.load_state_dict(ckpt["encoder_state_dict"])

    policy.to(device).eval()
    encoder.to(device).eval()
    return policy, encoder


def load_phase2(adapt_ckpt: str, cfg: dict, device: str):
    sd = torch.load(adapt_ckpt, map_location=device)
    a_cfg = sd["cfg"]
    adapt = Adaptation1DCNN(Adaptation1DCNNCfg(
        in_channels=a_cfg["in_channels"],
        history_length=a_cfg["history_length"],
        latent_dim=a_cfg["latent_dim"],
        embed_dim=a_cfg.get("embed_dim", 32),
    ))
    adapt.load_state_dict(sd["adaptation_state_dict"])
    adapt.to(device).eval()
    return adapt, a_cfg["history_length"], a_cfg["in_channels"]


# ──────────────────────────────────────────────────────────────
#  Per-trial runner (parameterized by mode)
# ──────────────────────────────────────────────────────────────

def run_trial(
    m_template, policy, encoder, adapt, cfg, trial: EvalTrial,
    mode: str, history_length: int, in_channels: int, device: str,
) -> EvalResult:
    """Run one MuJoCo trial under the given mode."""
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
    et_dim = int(cfg.get("rma_et_dim", 26))
    max_tau = 300.0

    d = mujoco.MjData(m_template)
    n_joints = d.qpos.shape[0] - 7

    torso_id = mujoco.mj_name2id(m_template, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    left_id = mujoco.mj_name2id(m_template, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
    right_id = mujoco.mj_name2id(m_template, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")

    default_arms = np.array(cfg.get("default_angles_arms",
                                     np.zeros(n_joints - n_leg)), dtype=np.float32)
    kps_arm = np.array(cfg.get("kps_arms",
                                np.ones(n_joints - n_leg) * 100), dtype=np.float32)
    kds_arm = np.array(cfg.get("kds_arms",
                                np.ones(n_joints - n_leg) * 5), dtype=np.float32)

    # Reset LSTM hidden state for the actor
    policy.memory_a.hidden_states = None

    action = np.zeros(n_leg, dtype=np.float32)
    counter = 0
    survival_time = eval_cfg["duration"]

    vx_errors, vy_errors, orientation_errors = [], [], []
    z_l2_to_teacher = []  # only for adaptation/baseline modes

    # History buffer for adaptation mode (rolling window of [obs, action])
    history = torch.zeros(1, history_length, in_channels, device=device)

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

        # Leg PD
        target_dof = action * cfg["action_scale"] + np.array(cfg["default_angles"][:n_leg], dtype=np.float32)
        leg_tau = pd_control(target_dof, d.qpos[7:7+n_leg], np.array(cfg["kps"], dtype=np.float32),
                             d.qvel[6:6+n_leg], np.array(cfg["kds"], dtype=np.float32))
        leg_tau = np.clip(np.nan_to_num(leg_tau), -max_tau, max_tau)
        d.ctrl[:n_leg] = leg_tau

        # Upper-body PD
        if n_joints > n_leg and d.ctrl.shape[0] > n_leg:
            n_upper = min(n_joints - n_leg, d.ctrl.shape[0] - n_leg)
            arm_tau = pd_control(default_arms[:n_upper], d.qpos[7+n_leg:7+n_leg+n_upper],
                                 kps_arm[:n_upper], d.qvel[6+n_leg:6+n_leg+n_upper], kds_arm[:n_upper])
            arm_tau = np.clip(np.nan_to_num(arm_tau), -max_tau, max_tau)
            d.ctrl[n_leg:n_leg+n_upper] = arm_tau

        mujoco.mj_step(m_template, d)
        counter += 1

        if d.qpos[2] < fall_height:
            survival_time = t
            break

        if counter % decim == 0:
            phase = (t / phase_period) % 1.0
            obs_47, proj_grav = compute_obs(d, cfg, action, cmd, phase, n_leg)

            # --- Build true e_t (always; needed by teacher and for z_l2 metric) ---
            if t >= force_start:
                forces = np.concatenate([trial.torso_force, trial.left_wrist_force,
                                         trial.right_wrist_force]).astype(np.float32)
            else:
                forces = np.zeros(9, dtype=np.float32)
            forces_norm = normalize_et_np(forces)
            if et_dim > 9:
                e_t_norm = np.concatenate([forces_norm, np.zeros(et_dim - 9, dtype=np.float32)])
            else:
                e_t_norm = forces_norm
            e_t_t = torch.from_numpy(e_t_norm).unsqueeze(0).float().to(device)

            with torch.no_grad():
                z_teacher = encoder(e_t_t)  # always compute, used as ground truth

                if mode == "teacher":
                    z_t = z_teacher
                elif mode == "adaptation":
                    z_t = adapt(history.reshape(1, -1))
                elif mode == "baseline":
                    z_t = torch.zeros(1, latent_dim, device=device)
                else:
                    raise ValueError(f"unknown mode: {mode}")

                if mode != "teacher":
                    z_l2_to_teacher.append(
                        torch.norm(z_t - z_teacher, dim=-1).item()
                    )

                actor_obs = torch.cat([
                    torch.from_numpy(obs_47).unsqueeze(0).to(device), z_t
                ], dim=-1).float()
                action_t = policy.act_inference(actor_obs)
                action = action_t.cpu().numpy().squeeze()

                # Push (obs, action) into rolling history (always; doesn't hurt)
                new_step = torch.cat([
                    torch.from_numpy(obs_47).unsqueeze(0).to(device),
                    action_t,
                ], dim=-1).unsqueeze(1)  # (1, 1, in_channels)
                history = torch.cat([history[:, 1:], new_step], dim=1)

            if t >= tracking_warmup:
                quat = d.qpos[3:7].copy()
                base_vel_world = d.qvel[0:3].copy()
                base_vel_body = quat_rotate_inverse(quat, base_vel_world)
                vx_errors.append((cmd[0] - base_vel_body[0]) ** 2)
                vy_errors.append((cmd[1] - base_vel_body[1]) ** 2)
                orientation_errors.append(np.linalg.norm(proj_grav[:2]))

    success = survival_time >= eval_cfg["duration"] - 0.01
    rmse_xy = float(np.sqrt(np.mean(np.array(vx_errors) + np.array(vy_errors)))) if vx_errors else float("nan")
    mean_orient = float(np.mean(orientation_errors)) if orientation_errors else float("nan")
    mean_zl2 = float(np.mean(z_l2_to_teacher)) if z_l2_to_teacher else 0.0

    total_mag = (np.linalg.norm(trial.torso_force) +
                 np.linalg.norm(trial.left_wrist_force) +
                 np.linalg.norm(trial.right_wrist_force))

    return EvalResult(
        label=trial.label, mode=mode,
        torso_force=trial.torso_force.tolist(),
        left_wrist_force=trial.left_wrist_force.tolist(),
        right_wrist_force=trial.right_wrist_force.tolist(),
        total_force_mag=float(total_mag),
        survival_time=round(float(survival_time), 3),
        success=bool(success),
        tracking_rmse_xy=round(rmse_xy, 4),
        mean_orientation_err=round(mean_orient, 4),
        mean_z_l2_to_teacher=round(mean_zl2, 4),
    )


# ──────────────────────────────────────────────────────────────
#  Trial generation (subset of sweep — keep eval focused)
# ──────────────────────────────────────────────────────────────

def generate_trials(cfg, magnitudes_override=None, sweep_mode_override=None) -> List[EvalTrial]:
    trials: List[EvalTrial] = []
    zero = np.zeros(3, dtype=np.float32)

    # Single-body sweep — axis (6 dirs) or spherical (N random dirs)
    magnitudes = magnitudes_override if magnitudes_override is not None \
        else cfg.get("phase2_magnitudes", [0, 5, 10, 15, 20, 25])
    mode = sweep_mode_override if sweep_mode_override is not None \
        else cfg.get("sweep_mode", "axis")
    sph_n = cfg.get("spherical_n_samples", 10)
    sph_seed = cfg.get("spherical_seed", 42)
    rng = np.random.default_rng(sph_seed)

    for body in cfg["sweep_bodies"]:
        bname = body["name"]
        blabel = body["label"]
        for mag in magnitudes:
            if mag == 0:
                trials.append(EvalTrial(
                    torso_force=zero.copy(), left_wrist_force=zero.copy(),
                    right_wrist_force=zero.copy(),
                    label=f"{blabel}|0N|zero",
                ))
                continue
            if mode == "axis":
                for dname, dvec in AXIS_DIRECTIONS.items():
                    force = dvec * mag
                    kwargs = {k: zero.copy() for k in BODY_KEY_MAP.values()}
                    kwargs[BODY_KEY_MAP[bname]] = force
                    trials.append(EvalTrial(**kwargs, label=f"{blabel}|{mag}N|{dname}"))
            elif mode == "spherical":
                dirs = sample_sphere(sph_n, rng)
                for i, dvec in enumerate(dirs):
                    force = (dvec * mag).astype(np.float32)
                    kwargs = {k: zero.copy() for k in BODY_KEY_MAP.values()}
                    kwargs[BODY_KEY_MAP[bname]] = force
                    trials.append(EvalTrial(**kwargs, label=f"{blabel}|{mag}N|s{i:02d}"))
            else:
                raise ValueError(f"Unknown sweep_mode: {mode}")

    # De-dup zero trials
    seen = set()
    out = []
    for t in trials:
        if t.label in seen:
            continue
        seen.add(t.label)
        out.append(t)
    return out


# ──────────────────────────────────────────────────────────────
#  CSV + summary
# ──────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "label", "mode",
    "torso_fx", "torso_fy", "torso_fz",
    "left_fx", "left_fy", "left_fz",
    "right_fx", "right_fy", "right_fz",
    "total_force_mag",
    "survival_time", "success",
    "tracking_rmse_xy", "mean_orientation_err",
    "mean_z_l2_to_teacher",
]


def result_to_row(r: EvalResult) -> dict:
    return {
        "label": r.label, "mode": r.mode,
        "torso_fx": r.torso_force[0], "torso_fy": r.torso_force[1], "torso_fz": r.torso_force[2],
        "left_fx": r.left_wrist_force[0], "left_fy": r.left_wrist_force[1], "left_fz": r.left_wrist_force[2],
        "right_fx": r.right_wrist_force[0], "right_fy": r.right_wrist_force[1], "right_fz": r.right_wrist_force[2],
        "total_force_mag": r.total_force_mag,
        "survival_time": r.survival_time, "success": r.success,
        "tracking_rmse_xy": r.tracking_rmse_xy,
        "mean_orientation_err": r.mean_orientation_err,
        "mean_z_l2_to_teacher": r.mean_z_l2_to_teacher,
    }


def write_csv(results: List[EvalResult], path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(result_to_row(r))


def print_summary(results: List[EvalResult]):
    print("\n" + "=" * 78)
    print(" PHASE 2 EVAL SUMMARY (per mode, all trials)")
    print("=" * 78)
    for mode in MODES:
        sub = [r for r in results if r.mode == mode]
        if not sub:
            continue
        n_total = len(sub)
        n_success = sum(1 for r in sub if r.success)
        rate = n_success / n_total * 100
        surv = [r.survival_time for r in sub]
        track = [r.tracking_rmse_xy for r in sub if r.success and not np.isnan(r.tracking_rmse_xy)]
        orient = [r.mean_orientation_err for r in sub if r.success and not np.isnan(r.mean_orientation_err)]
        zl2 = [r.mean_z_l2_to_teacher for r in sub if mode != "teacher"]

        print(f"\n  [{mode:>10s}]  {n_success}/{n_total} survived ({rate:.1f}%)")
        print(f"    Survival   : mean={np.mean(surv):.2f}s  min={np.min(surv):.2f}s")
        if track:
            print(f"    Track RMSE : mean={np.mean(track):.4f}  max={np.max(track):.4f} m/s")
        if orient:
            print(f"    Orient err : mean={np.mean(orient):.4f}")
        if zl2:
            print(f"    ||z - z_teacher||_2 : mean={np.mean(zl2):.4f}")

    # Per-magnitude breakdown
    print("\n" + "-" * 78)
    print(" Per-magnitude breakdown (success rate)")
    print("-" * 78)
    bodies = sorted({r.label.split("|")[0] for r in results})
    for body in bodies:
        print(f"\n  {body}:")
        body_rows = [r for r in results if r.label.startswith(body + "|")]
        mags = sorted({float(r.label.split("|")[1].replace("N", "")) for r in body_rows})
        header = "    mag | " + " | ".join(f"{m:>11s}" for m in MODES)
        print(header)
        print("    " + "-" * (len(header) - 4))
        for mag in mags:
            row = f"    {int(mag):3d}N|"
            for mode in MODES:
                msub = [r for r in body_rows
                        if r.mode == mode and float(r.label.split("|")[1].replace("N", "")) == mag]
                n_s = sum(1 for r in msub if r.success)
                n_t = len(msub)
                row += f"  {n_s:>3d}/{n_t:<3d} ({100*n_s/n_t if n_t else 0:5.1f}%)"
            print(row)
    print("\n" + "=" * 78)


def plot_results(csv_path: str, out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return

    bodies = sorted({r["label"].split("|")[0] for r in rows})
    mode_color = {"teacher": "C0", "adaptation": "C1", "baseline": "C3"}
    mode_marker = {"teacher": "o", "adaptation": "s", "baseline": "^"}

    # --- Plot 1: success rate vs magnitude (per body, all 3 modes) ---
    fig, axes = plt.subplots(1, len(bodies), figsize=(5 * len(bodies), 5), sharey=True)
    if len(bodies) == 1:
        axes = [axes]
    for ax, body in zip(axes, bodies):
        for mode in MODES:
            mode_rows = [r for r in rows
                         if r["mode"] == mode and r["label"].startswith(body + "|")]
            mag_succ = {}
            for r in mode_rows:
                mag = float(r["label"].split("|")[1].replace("N", ""))
                mag_succ.setdefault(mag, []).append(r["success"] == "True")
            if not mag_succ:
                continue
            mags_s = sorted(mag_succ.keys())
            rates = [100 * sum(mag_succ[m]) / len(mag_succ[m]) for m in mags_s]
            ax.plot(mags_s, rates, "-" + mode_marker[mode], color=mode_color[mode],
                    label=mode, markersize=7)
        ax.set_xlabel("Force magnitude (N)")
        ax.set_title(body)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Success rate (%)")
    fig.suptitle("Phase 2 eval: success rate vs perturbation (teacher / adaptation / baseline)")
    plt.tight_layout()
    p1 = os.path.join(out_dir, "phase2_success_vs_magnitude.png")
    plt.savefig(p1, dpi=150); plt.close()
    print(f"  Saved: {p1}")

    # --- Plot 2: tracking RMSE vs magnitude (survivors) ---
    fig, axes = plt.subplots(1, len(bodies), figsize=(5 * len(bodies), 5), sharey=True)
    if len(bodies) == 1:
        axes = [axes]
    for ax, body in zip(axes, bodies):
        for mode in MODES:
            mode_rows = [r for r in rows
                         if r["mode"] == mode and r["label"].startswith(body + "|")
                         and r["success"] == "True"]
            mag_track = {}
            for r in mode_rows:
                mag = float(r["label"].split("|")[1].replace("N", ""))
                val = float(r["tracking_rmse_xy"])
                if not np.isnan(val):
                    mag_track.setdefault(mag, []).append(val)
            if not mag_track:
                continue
            mags_s = sorted(mag_track.keys())
            means = [np.mean(mag_track[m]) for m in mags_s]
            ax.plot(mags_s, means, "-" + mode_marker[mode], color=mode_color[mode],
                    label=mode, markersize=7)
        ax.set_xlabel("Force magnitude (N)")
        ax.set_title(body)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Tracking RMSE (m/s)")
    fig.suptitle("Phase 2 eval: tracking RMSE vs perturbation (survivors)")
    plt.tight_layout()
    p2 = os.path.join(out_dir, "phase2_tracking_vs_magnitude.png")
    plt.savefig(p2, dpi=150); plt.close()
    print(f"  Saved: {p2}")

    # --- Plot 3: ||z - z_teacher||_2 over magnitudes (adaptation vs baseline) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in ("adaptation", "baseline"):
        mode_rows = [r for r in rows if r["mode"] == mode]
        mag_zl2 = {}
        for r in mode_rows:
            mag = float(r["label"].split("|")[1].replace("N", ""))
            mag_zl2.setdefault(mag, []).append(float(r["mean_z_l2_to_teacher"]))
        if not mag_zl2:
            continue
        mags_s = sorted(mag_zl2.keys())
        means = [np.mean(mag_zl2[m]) for m in mags_s]
        ax.plot(mags_s, means, "-" + mode_marker[mode], color=mode_color[mode],
                label=mode, markersize=7)
    ax.set_xlabel("Force magnitude (N)")
    ax.set_ylabel("||z - z_teacher||_2 (mean)")
    ax.set_title("Latent fidelity to teacher z_t")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    p3 = os.path.join(out_dir, "phase2_z_fidelity.png")
    plt.savefig(p3, dpi=150); plt.close()
    print(f"  Saved: {p3}")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RMA Phase 2 evaluation")
    parser.add_argument("--config", type=str, default=os.path.join(_SCRIPT_DIR, "sweep_config_m6000.yaml"))
    parser.add_argument("--policy_ckpt", type=str, required=True,
                        help="Phase 1 .pt with model_state_dict + encoder_state_dict")
    parser.add_argument("--adaptation_ckpt", type=str, required=True,
                        help="Phase 2 .pt from RmaPhase2Runner.save()")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="phase2_eval_results")
    parser.add_argument("--magnitudes", type=float, nargs="+", default=None,
                        help="Override force magnitudes (e.g. --magnitudes 20 30 40 50 60 70 80 90 100)")
    parser.add_argument("--sweep_mode", type=str, default=None, choices=["axis", "spherical"],
                        help="Override sweep_mode from config")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config_path = args.config if os.path.isabs(args.config) else os.path.join(_SCRIPT_DIR, args.config)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    config_dir = os.path.dirname(os.path.abspath(config_path))
    if "xml_path" in cfg and not os.path.isabs(cfg["xml_path"]):
        cfg["xml_path"] = os.path.normpath(os.path.join(config_dir, cfg["xml_path"]))
    for key in ("kps", "kds", "kps_arms", "kds_arms", "default_angles",
                "default_angles_arms", "cmd_scale"):
        if key in cfg:
            cfg[key] = np.array(cfg[key], dtype=np.float32)

    out_dir = os.path.join(_SCRIPT_DIR, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "phase2_eval.csv")

    # Load models
    policy_path = args.policy_ckpt if os.path.isabs(args.policy_ckpt) \
        else os.path.normpath(os.path.join(_SCRIPT_DIR, args.policy_ckpt))
    adapt_path = args.adaptation_ckpt if os.path.isabs(args.adaptation_ckpt) \
        else os.path.normpath(os.path.join(_SCRIPT_DIR, args.adaptation_ckpt))
    print(f"Phase 1 ckpt   : {policy_path}")
    print(f"Phase 2 ckpt   : {adapt_path}")

    policy, encoder = load_phase1(policy_path, cfg, device)
    adapt, history_length, in_channels = load_phase2(adapt_path, cfg, device)
    print(f"Adaptation cfg : history_length={history_length} in_channels={in_channels}")

    m = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    m.opt.timestep = cfg["simulation_dt"]

    trials = generate_trials(cfg, magnitudes_override=args.magnitudes,
                             sweep_mode_override=args.sweep_mode)
    print(f"\nTotal force trials: {len(trials)}  (× 3 modes = {len(trials)*3} runs)")
    print(f"Eval duration    : {cfg['eval']['duration']}s per run")
    est_min = len(trials) * 3 * cfg["eval"]["duration"] * 0.05 / 60  # crude estimate
    print(f"Estimated wall   : ~{est_min:.0f} min\n")

    # Run all (trial × mode)
    results: List[EvalResult] = []
    t0 = time.time()
    total = len(trials) * len(MODES)
    n_done = 0
    for trial in trials:
        for mode in MODES:
            r = run_trial(m, policy, encoder, adapt, cfg, trial, mode,
                          history_length, in_channels, device)
            results.append(r)
            n_done += 1
            elapsed = time.time() - t0
            eta = elapsed / n_done * (total - n_done)
            status = "OK" if r.success else f"FALL@{r.survival_time:.1f}s"
            print(f"  [{n_done:4d}/{total}] {r.mode:>10s} {status:>10s} | "
                  f"track={r.tracking_rmse_xy:.4f} | {r.label}  "
                  f"[{elapsed:.0f}s, eta={eta:.0f}s]")

    write_csv(results, csv_path)
    print(f"\nResults: {csv_path}")
    print_summary(results)
    plot_results(csv_path, out_dir)
    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
