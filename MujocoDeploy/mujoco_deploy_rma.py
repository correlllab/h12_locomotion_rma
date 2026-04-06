"""MuJoCo deployment for H1-2 RMA Phase 1 policy.

Loads the LSTM policy + encoder exported from Isaac Gym training,
applies configurable external forces to torso/wrists, and visualizes
the robot walking in MuJoCo.

Usage:
  # Viewer mode (default)
  python mujoco_deploy_rma.py --config h1_2_rma.yaml

  # With forces
  python mujoco_deploy_rma.py --config h1_2_rma.yaml \
      --torso_force 5 0 0 --left_force 0 0 -8 --right_force 0 0 -8

  # Headless (no viewer)
  python mujoco_deploy_rma.py --config h1_2_rma.yaml --no_view

  # Without encoder (naive baseline, z_t = 0)
  python mujoco_deploy_rma.py --config h1_2_rma.yaml --no_encode \
      --torso_force 5 0 0

  # Direct checkpoint loading (skip export step)
  python mujoco_deploy_rma.py --config h1_2_rma.yaml \
      --ckpt ../logs/h1_2_rma/<run>/model_5000.pt
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mujoco
import mujoco.viewer

from rsl_rl.modules import ActorCriticRecurrent
from rma.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    for key in ("kps", "kds", "kps_arms", "kds_arms",
                "default_angles", "default_angles_arms",
                "cmd_scale", "cmd_init",
                "torso_force", "left_wrist_force", "right_wrist_force"):
        if key in cfg:
            cfg[key] = np.array(cfg[key], dtype=np.float32)
    return cfg


def resolve_path(cfg, key, config_dir):
    """Resolve a config path relative to the YAML file location."""
    p = cfg.get(key, "")
    if p and not os.path.isabs(p):
        cfg[key] = os.path.normpath(os.path.join(config_dir, p))


def quat_rotate_inverse(q, v):
    """Rotate vector v by inverse of quaternion q  (MuJoCo w,x,y,z convention)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    # q_conj
    cw, cx, cy, cz = w, -x, -y, -z
    return np.array([
        v[0]*(cw**2+cx**2-cy**2-cz**2) + v[1]*2*(cx*cy-cw*cz) + v[2]*2*(cx*cz+cw*cy),
        v[0]*2*(cx*cy+cw*cz) + v[1]*(cw**2-cx**2+cy**2-cz**2) + v[2]*2*(cy*cz-cw*cx),
        v[0]*2*(cx*cz-cw*cy) + v[1]*2*(cy*cz+cw*cx) + v[2]*(cw**2-cx**2-cy**2+cz**2),
    ], dtype=np.float32)


def pd_control(target_q, q, kp, dq, kd):
    return (target_q - q) * kp - dq * kd


# ──────────────────────────────────────────────────────────────
#  Model loading
# ──────────────────────────────────────────────────────────────

def _remap_state_dict(model, state_dict):
    """Remap state dict keys to handle rsl_rl version differences.

    Training rsl_rl uses nn.Sequential directly:   actor.0.weight, actor.2.weight, ...
    Some rsl_rl versions wrap in SimpleMLP:         actor.layers.0.weight, ...
    This function adapts whichever direction is needed.
    """
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    # Check if remapping is needed
    if model_keys == ckpt_keys:
        return state_dict

    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        # ckpt has "actor.0.weight" but model wants "actor.layers.0.weight"
        if k.startswith("actor.") and "actor.layers." not in k and k.replace("actor.", "actor.layers.") in model_keys:
            new_k = k.replace("actor.", "actor.layers.", 1)
        elif k.startswith("critic.") and "critic.layers." not in k and k.replace("critic.", "critic.layers.") in model_keys:
            new_k = k.replace("critic.", "critic.layers.", 1)
        # ckpt has "actor.layers.0.weight" but model wants "actor.0.weight"
        elif "actor.layers." in k and k.replace("actor.layers.", "actor.") in model_keys:
            new_k = k.replace("actor.layers.", "actor.", 1)
        elif "critic.layers." in k and k.replace("critic.layers.", "critic.") in model_keys:
            new_k = k.replace("critic.layers.", "critic.", 1)
        remapped[new_k] = v

    mismatched = set(remapped.keys()) - model_keys
    if mismatched:
        print(f"Warning: still mismatched keys after remap: {mismatched}")
    else:
        print("  State dict keys remapped for rsl_rl version compatibility")
    return remapped


def load_policy_and_encoder(cfg, ckpt_path=None, device="cpu"):
    """Load ActorCriticRecurrent + EnvFactorEncoder.

    Either from exported files (policy.pt + encoder.pt)
    or directly from a training checkpoint (model_<iter>.pt).
    """
    if ckpt_path is not None:
        # Direct load from training checkpoint
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
        sd = _remap_state_dict(policy, ckpt["model_state_dict"])
        policy.load_state_dict(sd)

        enc_cfg = EnvFactorEncoderCfg(
            in_dim=cfg.get("rma_et_dim", 9),
            latent_dim=cfg.get("rma_latent_dim", 8),
            hidden_dims=(256, 128),
        )
        encoder = EnvFactorEncoder(enc_cfg)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        print(f"Loaded from checkpoint: {ckpt_path} (iter {ckpt.get('iter', '?')})")
    else:
        # Load from exported files
        pol_data = torch.load(cfg["policy_path"], map_location=device)
        policy = ActorCriticRecurrent(**pol_data["cfg"])
        sd = _remap_state_dict(policy, pol_data["model_state_dict"])
        policy.load_state_dict(sd)

        enc_data = torch.load(cfg["encoder_path"], map_location=device)
        encoder = EnvFactorEncoder(EnvFactorEncoderCfg(**enc_data["cfg"]))
        encoder.load_state_dict(enc_data["encoder_state_dict"])
        print(f"Loaded policy: {cfg['policy_path']}")
        print(f"Loaded encoder: {cfg['encoder_path']}")

    policy.eval()
    encoder.eval()
    return policy, encoder


# ──────────────────────────────────────────────────────────────
#  Observation computation (matches Isaac Gym h1_2_rma_env.py)
# ──────────────────────────────────────────────────────────────

def compute_obs(d, cfg, action, cmd, phase, n_leg_dofs=12):
    """Build the 47-dim observation vector matching Isaac Gym training.

    Layout:
      [0:3]   base_ang_vel * 0.25           (body frame)
      [3:6]   projected_gravity             (body frame)
      [6:9]   commands * cmd_scale
      [9:21]  (leg_dof_pos - default) * 1.0
      [21:33] leg_dof_vel * 0.05
      [33:45] last_actions
      [45]    sin(2*pi*phase)
      [46]    cos(2*pi*phase)
    """
    quat = d.qpos[3:7].copy()          # w, x, y, z
    omega = d.qvel[3:6].copy()          # ang vel already in body frame (MuJoCo free joint)
    qj = d.qpos[7:7+n_leg_dofs].copy()
    dqj = d.qvel[6:6+n_leg_dofs].copy()

    # MuJoCo d.qvel[3:6] is body-frame angular velocity (same as Isaac Gym's base_ang_vel)
    ang_vel_scaled = omega * cfg["ang_vel_scale"]
    # Gravity rotated into body frame
    projected_gravity = quat_rotate_inverse(quat, np.array([0., 0., -1.]))

    obs = np.zeros(47, dtype=np.float32)
    obs[0:3] = ang_vel_scaled
    obs[3:6] = projected_gravity
    obs[6:9] = cmd[:3] * cfg["cmd_scale"]
    obs[9:21] = (qj - cfg["default_angles"][:n_leg_dofs]) * cfg["dof_pos_scale"]
    obs[21:33] = dqj * cfg["dof_vel_scale"]
    obs[33:45] = action
    obs[45] = np.sin(2 * np.pi * phase)
    obs[46] = np.cos(2 * np.pi * phase)
    return obs


# ──────────────────────────────────────────────────────────────
#  Main simulation loop
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MuJoCo deploy for H1-2 RMA Phase 1")
    parser.add_argument("--config", type=str,
                        default=os.path.join(_SCRIPT_DIR, "h1_2_rma.yaml"))
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Direct training checkpoint (skips export)")
    parser.add_argument("--no_view", action="store_true",
                        help="Run headless")
    parser.add_argument("--no_encode", action="store_true",
                        help="Naive baseline: z_t = 0 (forces still applied in sim)")
    parser.add_argument("--torso_force", type=float, nargs=3, default=None,
                        help="Override torso force [Fx Fy Fz] in N")
    parser.add_argument("--left_force", type=float, nargs=3, default=None,
                        help="Override left wrist force [Fx Fy Fz] in N")
    parser.add_argument("--right_force", type=float, nargs=3, default=None,
                        help="Override right wrist force [Fx Fy Fz] in N")
    parser.add_argument("--cmd", type=float, nargs=3, default=None,
                        help="Override velocity command [vx vy yaw_rate]")
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    # Load config
    config_path = args.config if os.path.isabs(args.config) else os.path.join(_SCRIPT_DIR, args.config)
    cfg = load_config(config_path)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    for key in ("policy_path", "encoder_path", "xml_path"):
        resolve_path(cfg, key, config_dir)

    # CLI overrides
    if args.torso_force is not None:
        cfg["torso_force"] = np.array(args.torso_force, dtype=np.float32)
    if args.left_force is not None:
        cfg["left_wrist_force"] = np.array(args.left_force, dtype=np.float32)
    if args.right_force is not None:
        cfg["right_wrist_force"] = np.array(args.right_force, dtype=np.float32)
    if args.cmd is not None:
        cfg["cmd_init"] = np.array(args.cmd, dtype=np.float32)
    if args.duration is not None:
        cfg["simulation_duration"] = args.duration

    # Load models
    ckpt_path = args.ckpt
    if ckpt_path and not os.path.isabs(ckpt_path):
        ckpt_path = os.path.normpath(os.path.join(config_dir, ckpt_path))
    policy, encoder = load_policy_and_encoder(cfg, ckpt_path)

    use_encoder = not args.no_encode
    if args.no_encode:
        print(">> no_encode mode: z_t = 0 (forces still applied in sim)")

    latent_dim = cfg.get("rma_latent_dim", 8)

    # Load MuJoCo model
    m = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    d = mujoco.MjData(m)
    m.opt.timestep = cfg["simulation_dt"]

    n_joints = d.qpos.shape[0] - 7   # total DOFs (27 for handless)
    n_leg = cfg["num_actions"]        # 12
    print(f"MuJoCo model: {n_joints} DOFs, {n_leg} leg actions, ctrl size {d.ctrl.shape[0]}")

    # Find body indices for force application
    torso_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    left_wrist_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
    right_wrist_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    print(f"Force bodies: torso={torso_body_id}, left_wrist={left_wrist_body_id}, right_wrist={right_wrist_body_id}")

    # State
    action = np.zeros(n_leg, dtype=np.float32)
    cmd = cfg["cmd_init"].copy()
    phase_period = cfg.get("phase_period", 0.8)
    force_start = cfg.get("force_start_time", 0.0)
    decim = cfg["control_decimation"]
    dt = cfg["simulation_dt"]
    max_tau = 300.0

    # Upper-body defaults
    default_arms = cfg.get("default_angles_arms",
                           np.zeros(n_joints - n_leg, dtype=np.float32))
    kps_arm = cfg.get("kps_arms",
                      np.ones(n_joints - n_leg, dtype=np.float32) * 100.0)
    kds_arm = cfg.get("kds_arms",
                      np.ones(n_joints - n_leg, dtype=np.float32) * 5.0)

    duration = cfg["simulation_duration"]
    n_steps = int(duration / dt)
    counter = 0

    # Reset LSTM hidden state (starts as None, auto-inits on first forward)
    policy.memory_a.hidden_states = None

    print(f"\nForces (applied after t={force_start}s):")
    print(f"  torso:       {cfg['torso_force']}")
    print(f"  left_wrist:  {cfg['left_wrist_force']}")
    print(f"  right_wrist: {cfg['right_wrist_force']}")
    print(f"Commands: vx={cmd[0]:.2f} vy={cmd[1]:.2f} yaw={cmd[2]:.2f}")
    print(f"Duration: {duration}s | Headless: {args.no_view}\n")

    def step_fn(t):
        nonlocal action, counter

        # --- Apply external forces ---
        d.xfrc_applied[:] = 0
        if t >= force_start:
            if torso_body_id >= 0:
                d.xfrc_applied[torso_body_id, :3] = cfg["torso_force"]
            if left_wrist_body_id >= 0:
                d.xfrc_applied[left_wrist_body_id, :3] = cfg["left_wrist_force"]
            if right_wrist_body_id >= 0:
                d.xfrc_applied[right_wrist_body_id, :3] = cfg["right_wrist_force"]

        # --- PD control: legs ---
        target_dof = action * cfg["action_scale"] + cfg["default_angles"][:n_leg]
        leg_tau = pd_control(
            target_dof,
            d.qpos[7:7+n_leg],
            cfg["kps"],
            d.qvel[6:6+n_leg],
            cfg["kds"],
        )
        leg_tau = np.clip(np.nan_to_num(leg_tau), -max_tau, max_tau)
        d.ctrl[:n_leg] = leg_tau

        # --- PD control: upper body (hold at defaults) ---
        if n_joints > n_leg and d.ctrl.shape[0] > n_leg:
            n_upper = min(n_joints - n_leg, d.ctrl.shape[0] - n_leg)
            arm_tau = pd_control(
                default_arms[:n_upper],
                d.qpos[7+n_leg:7+n_leg+n_upper],
                kps_arm[:n_upper],
                d.qvel[6+n_leg:6+n_leg+n_upper],
                kds_arm[:n_upper],
            )
            arm_tau = np.clip(np.nan_to_num(arm_tau), -max_tau, max_tau)
            d.ctrl[n_leg:n_leg+n_upper] = arm_tau

        # --- Step physics ---
        mujoco.mj_step(m, d)
        counter += 1

        # --- Policy inference at decimation rate ---
        if counter % decim == 0:
            phase = (t / phase_period) % 1.0
            obs_47 = compute_obs(d, cfg, action, cmd, phase, n_leg)

            # Encode forces -> z_t
            if use_encoder and t >= force_start:
                e_t = np.concatenate([
                    cfg["torso_force"],
                    cfg["left_wrist_force"],
                    cfg["right_wrist_force"],
                ]).astype(np.float32)
            else:
                e_t = np.zeros(cfg.get("rma_et_dim", 9), dtype=np.float32)

            with torch.no_grad():
                e_t_tensor = torch.from_numpy(e_t).unsqueeze(0).float()
                z_t = encoder(e_t_tensor).numpy().squeeze()  # (8,)

            # Concatenate obs + z_t -> 55
            actor_obs = np.concatenate([obs_47, z_t]).astype(np.float32)
            obs_tensor = torch.from_numpy(actor_obs).unsqueeze(0).float()

            with torch.no_grad():
                action = policy.act_inference(obs_tensor).numpy().squeeze()

            # Periodic logging
            if counter % (decim * 100) == 0:
                h = d.qpos[2]
                print(f"  t={t:6.2f}s | height={h:.3f} | "
                      f"action [{action[:3].round(3)}...] | "
                      f"z_t [{z_t[:3].round(3)}...]")

    # --- Run ---
    if not args.no_view:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            print("Viewer open. Close window to stop.")
            for step in range(n_steps):
                t = step * dt
                step_fn(t)
                if not viewer.is_running():
                    break
                viewer.sync()
                # Real-time pacing
                sleep_t = dt - (time.time() % dt)
                if sleep_t > 0:
                    time.sleep(sleep_t)
    else:
        for step in range(n_steps):
            step_fn(step * dt)

    print("\nDone.")


if __name__ == "__main__":
    main()
