"""Headless probe: try a few multi-body force patterns across seeds, report
survival per (teacher / adaptation / baseline) mode. Used to pick the exact
(pattern, seed) tuples that cleanly show 'adaptation saves the robot where
baseline falls'.

Usage:
  python MujocoDeploy/probe_video_seeds.py \
      --policy_ckpt logs/.../model_6000.pt \
      --adaptation_ckpt logs/.../adaptation_5000.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

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
    compute_obs, _remap_state_dict, normalize_et_np, pd_control, quat_rotate_inverse,
)


MODES = ("teacher", "adaptation", "baseline")


@dataclass
class Pattern:
    label: str
    torso: np.ndarray    # (3,) N
    left: np.ndarray     # (3,) N
    right: np.ndarray    # (3,) N


def load_phase1(policy_ckpt, cfg, device):
    ckpt = torch.load(policy_ckpt, map_location=device)
    latent_dim = cfg.get("rma_latent_dim", 8)
    et_dim = cfg.get("rma_et_dim", 9)
    policy = ActorCriticRecurrent(
        num_actor_obs=47 + latent_dim, num_critic_obs=50 + latent_dim,
        num_actions=cfg["num_actions"],
        actor_hidden_dims=[32], critic_hidden_dims=[32],
        rnn_type="lstm", rnn_hidden_size=64, rnn_num_layers=1,
        activation="elu",
    )
    policy.load_state_dict(_remap_state_dict(policy, ckpt["model_state_dict"]))
    encoder = EnvFactorEncoder(EnvFactorEncoderCfg(
        in_dim=et_dim, latent_dim=latent_dim, hidden_dims=(256, 128),
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


def run_headless(m, policy, encoder, adapt, cfg, pat: Pattern,
                 seed: int, mode: str, hl: int, ic: int, device: str,
                 duration: float, force_start: float):
    dt = cfg["simulation_dt"]
    decim = cfg["control_decimation"]
    n_leg = cfg["num_actions"]
    n_steps = int(duration / dt)
    fall_height = 0.5
    cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    phase_period = cfg.get("phase_period", 0.8)
    latent_dim = cfg.get("rma_latent_dim", 8)
    et_dim = int(cfg.get("rma_et_dim", 9))
    max_tau = 300.0

    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    n_joints = d.qpos.shape[0] - 7
    default_legs = np.array(cfg["default_angles"][:n_leg], dtype=np.float32)
    default_arms = np.array(cfg.get("default_angles_arms",
                                     np.zeros(n_joints - n_leg)), dtype=np.float32)
    jitter = rng.normal(0, 0.02, size=n_leg).astype(np.float32)
    d.qpos[7:7+n_leg] = default_legs + jitter
    if n_joints > n_leg:
        d.qpos[7+n_leg:7+n_joints] = default_arms[:n_joints - n_leg]

    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    left_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
    right_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    kps_leg = np.array(cfg["kps"], dtype=np.float32)
    kds_leg = np.array(cfg["kds"], dtype=np.float32)
    kps_arm = np.array(cfg.get("kps_arms", np.ones(n_joints - n_leg) * 100), dtype=np.float32)
    kds_arm = np.array(cfg.get("kds_arms", np.ones(n_joints - n_leg) * 5), dtype=np.float32)

    policy.memory_a.hidden_states = None
    action = np.zeros(n_leg, dtype=np.float32)
    counter = 0
    survival_time = duration
    history = torch.zeros(1, hl, ic, device=device)

    for step in range(n_steps):
        t = step * dt
        d.xfrc_applied[:] = 0
        if t >= force_start:
            if torso_id >= 0: d.xfrc_applied[torso_id, :3] = pat.torso
            if left_id >= 0:  d.xfrc_applied[left_id,  :3] = pat.left
            if right_id >= 0: d.xfrc_applied[right_id, :3] = pat.right

        target_dof = action * cfg["action_scale"] + default_legs
        leg_tau = pd_control(target_dof, d.qpos[7:7+n_leg], kps_leg,
                             d.qvel[6:6+n_leg], kds_leg)
        leg_tau = np.clip(np.nan_to_num(leg_tau), -max_tau, max_tau)
        d.ctrl[:n_leg] = leg_tau
        if n_joints > n_leg and d.ctrl.shape[0] > n_leg:
            nu = min(n_joints - n_leg, d.ctrl.shape[0] - n_leg)
            arm_tau = pd_control(default_arms[:nu], d.qpos[7+n_leg:7+n_leg+nu],
                                 kps_arm[:nu], d.qvel[6+n_leg:6+n_leg+nu], kds_arm[:nu])
            arm_tau = np.clip(np.nan_to_num(arm_tau), -max_tau, max_tau)
            d.ctrl[n_leg:n_leg+nu] = arm_tau
        mujoco.mj_step(m, d)
        counter += 1
        if d.qpos[2] < fall_height:
            survival_time = t
            break
        if counter % decim == 0:
            phase = (t / phase_period) % 1.0
            obs_47, _ = compute_obs(d, cfg, action, cmd, phase, n_leg)
            if t >= force_start:
                forces = np.concatenate([pat.torso, pat.left, pat.right]).astype(np.float32)
            else:
                forces = np.zeros(9, dtype=np.float32)
            e_t_norm = normalize_et_np(forces)
            if et_dim > 9:
                e_t_norm = np.concatenate([e_t_norm, np.zeros(et_dim - 9, dtype=np.float32)])
            e_t_t = torch.from_numpy(e_t_norm).unsqueeze(0).float().to(device)
            with torch.no_grad():
                z_teacher = encoder(e_t_t)
                if mode == "teacher":     z_t = z_teacher
                elif mode == "adaptation": z_t = adapt(history.reshape(1, -1))
                else:                      z_t = torch.zeros(1, latent_dim, device=device)
                actor_obs = torch.cat([
                    torch.from_numpy(obs_47).unsqueeze(0).to(device), z_t
                ], dim=-1).float()
                action_t = policy.act_inference(actor_obs)
                action = action_t.cpu().numpy().squeeze()
                new_step = torch.cat([
                    torch.from_numpy(obs_47).unsqueeze(0).to(device), action_t,
                ], dim=-1).unsqueeze(1)
                history = torch.cat([history[:, 1:], new_step], dim=1)

    return survival_time, survival_time >= duration - 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_ckpt", required=True)
    ap.add_argument("--adaptation_ckpt", required=True)
    ap.add_argument("--config", default=os.path.join(_SCRIPT_DIR, "sweep_config_m6000.yaml"))
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--force_start", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    if "xml_path" in cfg and not os.path.isabs(cfg["xml_path"]):
        cfg["xml_path"] = os.path.normpath(os.path.join(cfg_dir, cfg["xml_path"]))
    for key in ("kps", "kds", "kps_arms", "kds_arms", "default_angles",
                "default_angles_arms", "cmd_scale"):
        if key in cfg:
            cfg[key] = np.array(cfg[key], dtype=np.float32)

    policy, encoder = load_phase1(args.policy_ckpt, cfg, device)
    adapt, hl, ic = load_phase2(args.adaptation_ckpt, device)
    m = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    m.opt.timestep = cfg["simulation_dt"]

    # Candidate multi-body patterns. All scale-up variations of the conditions
    # that showed the largest OOD adaptation lift in the stress eval:
    # aligned wrists, and combined (torso + both wrists in same direction).
    patterns: List[Pattern] = [
        # Aligned wrists (both +X): emulates carrying a payload forward.
        Pattern("aligned_80+X", np.zeros(3,np.float32),
                np.array([80,0,0],np.float32), np.array([80,0,0],np.float32)),
        Pattern("aligned_100+X", np.zeros(3,np.float32),
                np.array([100,0,0],np.float32), np.array([100,0,0],np.float32)),
        Pattern("aligned_80+Y", np.zeros(3,np.float32),
                np.array([0,80,0],np.float32), np.array([0,80,0],np.float32)),
        # Combined (torso + both wrists same direction): body pushed through payload.
        Pattern("combined_80_60_60+X", np.array([80,0,0],np.float32),
                np.array([60,0,0],np.float32), np.array([60,0,0],np.float32)),
        Pattern("combined_60_50_50+X", np.array([60,0,0],np.float32),
                np.array([50,0,0],np.float32), np.array([50,0,0],np.float32)),
        Pattern("combined_50_40_40+Y", np.array([0,50,0],np.float32),
                np.array([0,40,0],np.float32), np.array([0,40,0],np.float32)),
    ]

    print(f"{'pattern':<30s} {'seed':>4s}   {'T':>6s}  {'A':>6s}  {'B':>6s}   flags")
    print("-" * 80)
    hits: List[Tuple[str, int, dict]] = []
    for pat in patterns:
        for seed in args.seeds:
            results = {}
            for mode in MODES:
                surv, succ = run_headless(m, policy, encoder, adapt, cfg, pat,
                                          seed, mode, hl, ic, device,
                                          args.duration, args.force_start)
                results[mode] = {"surv": surv, "succ": succ}
            T, A, B = results["teacher"], results["adaptation"], results["baseline"]
            flag = ""
            # Prefer: adaptation saves where baseline falls (clean story).
            if A["succ"] and not B["succ"]:
                flag = "★ A>B"
                hits.append((pat.label, seed, results))
            elif not A["succ"] and not B["succ"] and A["surv"] - B["surv"] > 1.0:
                flag = "• A>B (both fell, A lasted +1s)"
                hits.append((pat.label, seed, results))
            print(f"{pat.label:<30s} {seed:>4d}   "
                  f"{T['surv']:>5.2f}s {A['surv']:>5.2f}s {B['surv']:>5.2f}s   {flag}")
        print()

    print("=" * 80)
    print(f"Clean hits (A saves where B falls, or A lasts ≥1s longer): {len(hits)}")
    for label, seed, r in hits:
        print(f"  {label:<30s} seed={seed}  T={r['teacher']['surv']:.1f}  "
              f"A={r['adaptation']['surv']:.1f}  B={r['baseline']['surv']:.1f}")


if __name__ == "__main__":
    main()
