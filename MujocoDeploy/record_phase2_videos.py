"""Record paired teacher / adaptation / baseline videos for Phase 2 eval.

Usage:
  python MujocoDeploy/record_phase2_videos.py \
      --policy_ckpt logs/h1_2_rma/Apr19_14-17-35_paperfull_v1/model_6000.pt \
      --adaptation_ckpt logs/h1_2_rma_phase2/Apr19_23-29-32_phase2_paperfix/adaptation_1000.pt \
      --conditions left_wrist:90:+Y:walk:0
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime

import cv2
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
from MujocoDeploy.comprehensive_eval import COMMANDS
from MujocoDeploy.sweep_rma_forces import (
    AXIS_DIRECTIONS, BODY_KEY_MAP, compute_obs,
    _remap_state_dict, normalize_et_np, pd_control,
)
from MujocoDeploy.record_rma_videos import (
    draw_overlay, write_mp4, stitch_side_by_side, build_condition,
    parse_condition,
)


MODES = ("teacher", "adaptation", "baseline")
MODE_COLOR_BGR = {
    "teacher":    (60, 180, 60),
    "adaptation": (200, 150, 40),
    "baseline":   (60, 60, 220),
}


def load_phase1(policy_ckpt, cfg, device):
    ckpt = torch.load(policy_ckpt, map_location=device)
    latent_dim = cfg.get("rma_latent_dim", 8)
    et_dim = cfg.get("rma_et_dim", 26)
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


def run_trial_render(m, policy, encoder, adapt, cfg, cond, mode,
                     history_length, in_channels, device, renderer,
                     render_fps=30):
    eval_cfg = cfg["eval"]
    dt = cfg["simulation_dt"]
    decim = cfg["control_decimation"]
    n_leg = cfg["num_actions"]
    n_steps = int(eval_cfg["duration"] / dt)
    fall_height = eval_cfg["fall_height"]
    force_start = eval_cfg.get("force_start_time", 1.0)
    phase_period = cfg.get("phase_period", 0.8)
    et_dim = int(cfg.get("rma_et_dim", 26))
    latent_dim = cfg.get("rma_latent_dim", 8)
    steps_per_frame = max(1, int(round(1.0 / (render_fps * dt))))

    d = mujoco.MjData(m)
    rng = np.random.default_rng(cond.seed)
    n_joints = d.qpos.shape[0] - 7
    default_legs = np.array(cfg["default_angles"][:n_leg], dtype=np.float32)
    default_arms = np.array(cfg.get("default_angles_arms",
                                     np.zeros(n_joints - n_leg)), dtype=np.float32)
    jitter = rng.normal(0, 0.02, size=n_leg).astype(np.float32)
    d.qpos[7:7+n_leg] = default_legs + jitter

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
    survival_time = eval_cfg["duration"]
    frames = []
    history = torch.zeros(1, history_length, in_channels, device=device)

    cmd = COMMANDS[cond.command_tag].astype(np.float32)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = torso_id
    cam.distance = 3.5
    cam.azimuth = 130
    cam.elevation = -15
    cam.lookat[:] = d.qpos[0:3]

    for step in range(n_steps):
        t = step * dt
        d.xfrc_applied[:] = 0
        if t >= force_start:
            if torso_id >= 0:
                d.xfrc_applied[torso_id, :3] = cond.torso_force
            if left_id >= 0:
                d.xfrc_applied[left_id, :3] = cond.left_wrist_force
            if right_id >= 0:
                d.xfrc_applied[right_id, :3] = cond.right_wrist_force

        target_dof = action * cfg["action_scale"] + default_legs
        leg_tau = pd_control(target_dof, d.qpos[7:7+n_leg], kps_leg,
                             d.qvel[6:6+n_leg], kds_leg)
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
            survival_time = t
            for _ in range(int(render_fps * 0.3)):
                mujoco.mj_step(m, d)
                renderer.update_scene(d, cam)
                frames.append(_annotate(renderer.render(), cond, mode, t, fallen=True))
            break

        if counter % decim == 0:
            phase = (t / phase_period) % 1.0
            obs_47, _ = compute_obs(d, cfg, action, cmd, phase, n_leg)

            forces = np.zeros(9, dtype=np.float32)
            if t >= force_start:
                forces[:3] = cond.torso_force
                forces[3:6] = cond.left_wrist_force
                forces[6:9] = cond.right_wrist_force
            forces_norm = normalize_et_np(forces)
            e_t_norm = np.concatenate([forces_norm,
                                        np.zeros(et_dim - 9, dtype=np.float32)]) \
                if et_dim > 9 else forces_norm
            e_t_t = torch.from_numpy(e_t_norm).unsqueeze(0).float().to(device)

            with torch.no_grad():
                if mode == "teacher":
                    z_t = encoder(e_t_t)
                elif mode == "adaptation":
                    z_t = adapt(history.reshape(1, -1))
                else:
                    z_t = torch.zeros(1, latent_dim, device=device)

                actor_obs = torch.cat([
                    torch.from_numpy(obs_47).unsqueeze(0).to(device), z_t
                ], dim=-1).float()
                action_t = policy.act_inference(actor_obs)
                action = action_t.cpu().numpy().squeeze()

                new_step = torch.cat([
                    torch.from_numpy(obs_47).unsqueeze(0).to(device), action_t,
                ], dim=-1).unsqueeze(1)
                history = torch.cat([history[:, 1:], new_step], dim=1)

        if counter % steps_per_frame == 0:
            renderer.update_scene(d, cam)
            frames.append(_annotate(renderer.render(), cond, mode, t, fallen=False))

    return frames, {
        "survival_time": round(survival_time, 3),
        "success": survival_time >= eval_cfg["duration"] - 0.01,
        "mode": mode,
    }


def _annotate(frame_rgb, cond, mode, t, fallen):
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    color = MODE_COLOR_BGR[mode]
    lines = [
        f"{mode.upper()}",
        f"{cond.body}  |  {int(cond.magnitude)}N {cond.direction_tag}",
        f"cmd={cond.command_tag}  seed={cond.seed}",
        f"t={t:5.2f}s" + ("   FALLEN" if fallen else ""),
    ]
    return draw_overlay(frame_bgr, lines, color=color)


def stitch_triple(frames_list, labels):
    n = max(len(f) for f in frames_list)
    def pad(fs):
        if not fs:
            return []
        return fs + [fs[-1]] * (n - len(fs))
    padded = [pad(f) for f in frames_list]
    out = []
    for tup in zip(*padded):
        h = tup[0].shape[0]
        dividers = [np.full((h, 4, 3), 255, dtype=np.uint8) for _ in range(len(tup)-1)]
        parts = []
        for i, f in enumerate(tup):
            parts.append(f)
            if i < len(tup) - 1:
                parts.append(dividers[i])
        out.append(np.hstack(parts))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_ckpt", type=str, required=True)
    parser.add_argument("--adaptation_ckpt", type=str, required=True)
    parser.add_argument("--config", type=str,
                        default=os.path.join(_SCRIPT_DIR, "sweep_config_m6000.yaml"))
    parser.add_argument("--tag", type=str, default="phase2_paperfix")
    parser.add_argument("--out_root", type=str,
                        default=os.path.join(_SCRIPT_DIR, "eval_results/phase2_paperfix_stress/videos"))
    parser.add_argument("--conditions", type=str, nargs="+", required=True,
                        help="body:mag:dir:cmd:seed (e.g. left_wrist:90:+Y:walk:0)")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--force_start", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

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
    cfg.setdefault("eval", {})
    cfg["eval"]["duration"] = args.duration
    cfg["eval"].setdefault("fall_height", 0.5)
    cfg["eval"]["force_start_time"] = args.force_start

    conditions = [parse_condition(s) for s in args.conditions]

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir = os.path.join(args.out_root, f"{args.tag}__{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    policy, encoder = load_phase1(args.policy_ckpt, cfg, device)
    adapt, hl, ic = load_phase2(args.adaptation_ckpt, device)
    print(f"Loaded policy + adaptation (hl={hl}, in_ch={ic})")

    m = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    m.opt.timestep = cfg["simulation_dt"]
    renderer = mujoco.Renderer(m, height=args.height, width=args.width)

    summary = ["# Phase 2 paired videos", ""]
    for cond in conditions:
        print(f"\n=== {cond.name} ===")
        per_mode_frames = []
        per_mode_meta = []
        for mode in MODES:
            print(f"  running {mode} …", end=" ", flush=True)
            frames, meta = run_trial_render(
                m, policy, encoder, adapt, cfg, cond, mode,
                hl, ic, device, renderer, render_fps=args.fps,
            )
            print(f"surv={meta['survival_time']}s success={meta['success']}")
            per_mode_frames.append(frames)
            per_mode_meta.append(meta)
            path = os.path.join(out_dir, f"{cond.name}__{mode}.mp4")
            write_mp4(frames, path, fps=args.fps)

        # Stitch all three
        triple = stitch_triple(per_mode_frames, MODES)
        path = os.path.join(out_dir, f"{cond.name}__triple.mp4")
        write_mp4(triple, path, fps=args.fps)

        summary.append(f"## {cond.name}")
        for mode, meta in zip(MODES, per_mode_meta):
            summary.append(f"- **{mode}**: surv={meta['survival_time']}s  "
                           f"success={meta['success']}")
        summary.append("")

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(summary))
    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
