"""Diagnose why Phase 2 adaptation is worse than baseline z=0 in MuJoCo.

Hypothesis: in MuJoCo, mass/COM/motor/friction are nominal — only the 9 force
dims of e_t are non-zero. The encoder's "true z_t" lives close to encoder(0).
- baseline z=0 may already be close to that
- adaptation z, trained on IsaacGym distribution where ALL 26 dims vary, has
  never seen this degenerate input distribution and produces OOD predictions

This script runs ONE force trial under each of {teacher, adaptation, baseline}
and dumps per-step z_t values + distances for inspection.
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml
import mujoco

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rsl_rl.modules import ActorCriticRecurrent
from rma.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg
from rma.adaptation_module import Adaptation1DCNN, Adaptation1DCNNCfg
from MujocoDeploy.sweep_rma_forces import (
    quat_rotate_inverse, pd_control, compute_obs,
    _remap_state_dict, normalize_et_np,
)


def load_phase1(policy_ckpt, cfg, device):
    ckpt = torch.load(policy_ckpt, map_location=device)
    policy = ActorCriticRecurrent(
        num_actor_obs=47 + cfg.get("rma_latent_dim", 8),
        num_critic_obs=50 + cfg.get("rma_latent_dim", 8),
        num_actions=cfg["num_actions"],
        actor_hidden_dims=[32], critic_hidden_dims=[32],
        rnn_type="lstm", rnn_hidden_size=64, rnn_num_layers=1,
        activation="elu",
    )
    policy.load_state_dict(_remap_state_dict(policy, ckpt["model_state_dict"]))
    encoder = EnvFactorEncoder(EnvFactorEncoderCfg(
        in_dim=cfg.get("rma_et_dim", 26),
        latent_dim=cfg.get("rma_latent_dim", 8),
        hidden_dims=(256, 128),
    ))
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    return policy.to(device).eval(), encoder.to(device).eval()


def load_phase2(adapt_ckpt, device):
    sd = torch.load(adapt_ckpt, map_location=device)
    a = sd["cfg"]
    adapt = Adaptation1DCNN(Adaptation1DCNNCfg(
        in_channels=a["in_channels"], history_length=a["history_length"],
        latent_dim=a["latent_dim"], embed_dim=a.get("embed_dim", 32),
    ))
    adapt.load_state_dict(sd["adaptation_state_dict"])
    return adapt.to(device).eval(), a["history_length"], a["in_channels"]


def run_and_log(m, policy, encoder, adapt, cfg, mode, force_vec, device,
                history_length, in_channels):
    """Run one trial, log z_teacher, z_pred, ||z_pred - z_teacher|| per step."""
    eval_cfg = cfg["eval"]
    dt = cfg["simulation_dt"]
    decim = cfg["control_decimation"]
    n_leg = cfg["num_actions"]
    n_steps = int(eval_cfg["duration"] / dt)
    fall_height = eval_cfg["fall_height"]
    force_start = eval_cfg.get("force_start_time", 1.0)
    cmd = np.array(eval_cfg["cmd"], dtype=np.float32)
    phase_period = cfg.get("phase_period", 0.8)
    et_dim = int(cfg.get("rma_et_dim", 26))
    latent_dim = cfg.get("rma_latent_dim", 8)

    d = mujoco.MjData(m)
    n_joints = d.qpos.shape[0] - 7

    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    left_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
    default_arms = np.array(cfg.get("default_angles_arms",
                                     np.zeros(n_joints - n_leg)), dtype=np.float32)
    kps_arm = np.array(cfg.get("kps_arms", np.ones(n_joints - n_leg)*100), dtype=np.float32)
    kds_arm = np.array(cfg.get("kds_arms", np.ones(n_joints - n_leg)*5), dtype=np.float32)

    policy.memory_a.hidden_states = None
    action = np.zeros(n_leg, dtype=np.float32)
    counter = 0

    history = torch.zeros(1, history_length, in_channels, device=device)
    log = {"t": [], "z_teacher": [], "z_pred": [], "z_l2": []}

    survival = eval_cfg["duration"]
    for step in range(n_steps):
        t = step * dt

        d.xfrc_applied[:] = 0
        if t >= force_start:
            if torso_id >= 0:
                d.xfrc_applied[torso_id, :3] = force_vec[0]
            if left_id >= 0:
                d.xfrc_applied[left_id, :3] = force_vec[1]

        target_dof = action * cfg["action_scale"] + np.array(cfg["default_angles"][:n_leg], dtype=np.float32)
        leg_tau = pd_control(target_dof, d.qpos[7:7+n_leg],
                             np.array(cfg["kps"], dtype=np.float32),
                             d.qvel[6:6+n_leg],
                             np.array(cfg["kds"], dtype=np.float32))
        d.ctrl[:n_leg] = np.clip(np.nan_to_num(leg_tau), -300, 300)
        if n_joints > n_leg and d.ctrl.shape[0] > n_leg:
            n_upper = min(n_joints - n_leg, d.ctrl.shape[0] - n_leg)
            arm_tau = pd_control(default_arms[:n_upper],
                                 d.qpos[7+n_leg:7+n_leg+n_upper],
                                 kps_arm[:n_upper],
                                 d.qvel[6+n_leg:6+n_leg+n_upper], kds_arm[:n_upper])
            d.ctrl[n_leg:n_leg+n_upper] = np.clip(np.nan_to_num(arm_tau), -300, 300)

        mujoco.mj_step(m, d)
        counter += 1
        if d.qpos[2] < fall_height:
            survival = t
            break

        if counter % decim == 0:
            phase = (t / phase_period) % 1.0
            obs_47, _ = compute_obs(d, cfg, action, cmd, phase, n_leg)

            forces = np.zeros(9, dtype=np.float32)
            if t >= force_start:
                forces[:3] = force_vec[0]
                forces[3:6] = force_vec[1]
            forces_norm = normalize_et_np(forces)
            e_t_norm = np.concatenate([forces_norm, np.zeros(et_dim - 9, dtype=np.float32)]) \
                if et_dim > 9 else forces_norm
            e_t_t = torch.from_numpy(e_t_norm).unsqueeze(0).float().to(device)

            with torch.no_grad():
                z_teacher = encoder(e_t_t)

                if mode == "teacher":
                    z_t = z_teacher
                elif mode == "adaptation":
                    z_t = adapt(history.reshape(1, -1))
                else:  # baseline
                    z_t = torch.zeros(1, latent_dim, device=device)

                actor_obs = torch.cat([
                    torch.from_numpy(obs_47).unsqueeze(0).to(device), z_t
                ], dim=-1).float()
                action_t = policy.act_inference(actor_obs)
                action = action_t.cpu().numpy().squeeze()

                # Log z values
                log["t"].append(t)
                log["z_teacher"].append(z_teacher.cpu().numpy().squeeze())
                log["z_pred"].append(z_t.cpu().numpy().squeeze())
                log["z_l2"].append(float(torch.norm(z_t - z_teacher).item()))

                # Update history
                new_step = torch.cat([
                    torch.from_numpy(obs_47).unsqueeze(0).to(device),
                    action_t,
                ], dim=-1).unsqueeze(1)
                history = torch.cat([history[:, 1:], new_step], dim=1)

    return survival, log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=os.path.join(_SCRIPT_DIR, "sweep_config_m6000.yaml"))
    p.add_argument("--policy_ckpt", type=str, required=True)
    p.add_argument("--adaptation_ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--torso_force", type=float, nargs=3, default=[0, 0, 0])
    p.add_argument("--left_wrist_force", type=float, nargs=3, default=[0, 90, 0])
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    if "xml_path" in cfg and not os.path.isabs(cfg["xml_path"]):
        cfg["xml_path"] = os.path.normpath(os.path.join(cfg_dir, cfg["xml_path"]))
    for key in ("kps", "kds", "kps_arms", "kds_arms", "default_angles",
                "default_angles_arms", "cmd_scale"):
        if key in cfg:
            cfg[key] = np.array(cfg[key], dtype=np.float32)

    policy, encoder = load_phase1(args.policy_ckpt, cfg, args.device)
    adapt, hl, ic = load_phase2(args.adaptation_ckpt, args.device)

    m = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    m.opt.timestep = cfg["simulation_dt"]

    force_vec = (np.array(args.torso_force, dtype=np.float32),
                 np.array(args.left_wrist_force, dtype=np.float32))
    print(f"Forces: torso={force_vec[0]}, left_wrist={force_vec[1]}")

    # Compute encoder(0) baseline reference
    with torch.no_grad():
        z_at_zero = encoder(torch.zeros(1, cfg.get("rma_et_dim", 26), device=args.device))
    print(f"\nencoder(0_input) = {z_at_zero.cpu().numpy().squeeze()}")
    print(f"||encoder(0)|| = {torch.norm(z_at_zero).item():.4f}")
    print(f"  → if this is ~0, baseline z=0 is correct in nominal env;")
    print(f"  → if non-zero, both adaptation and baseline are wrong.")

    print("\n" + "=" * 70)
    for mode in ("teacher", "adaptation", "baseline"):
        surv, log = run_and_log(m, policy, encoder, adapt, cfg, mode, force_vec,
                                args.device, hl, ic)
        z_l2 = np.array(log["z_l2"])
        z_pred = np.array(log["z_pred"])
        z_teach = np.array(log["z_teacher"])
        # Pre-force vs post-force stats
        t = np.array(log["t"])
        force_start = cfg["eval"].get("force_start_time", 1.0)
        pre = t < force_start
        post = t >= force_start
        print(f"\n[{mode:>10s}] survival={surv:.2f}s")
        print(f"  ||z_pred - z_teacher||  pre-force(mean)={z_l2[pre].mean():.4f}  post(mean)={z_l2[post].mean():.4f}")
        print(f"  z_pred  pre-force range=[{z_pred[pre].min():.3f},{z_pred[pre].max():.3f}]  post=[{z_pred[post].min():.3f},{z_pred[post].max():.3f}]")
        print(f"  z_teacher post mean={z_teach[post].mean(axis=0).round(3)}")
        print(f"  z_pred    post mean={z_pred[post].mean(axis=0).round(3)}")


if __name__ == "__main__":
    main()
