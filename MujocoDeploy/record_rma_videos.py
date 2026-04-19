"""Record paired RMA vs baseline comparison videos for force-perturbation failures.

For each condition (body, force, command, seed) we run the policy twice —
once with the RMA encoder seeing e_t, once with z_t = encoder(0) — while
capturing frames from a tracking MuJoCo camera.  The two runs are stitched
side-by-side into a single MP4 per condition, plus individual MP4s.

Reuses the simulator helpers from `comprehensive_eval.run_trial` but with
a Renderer hooked in; state is otherwise identical so videos correspond
exactly to the conditions surfaced in `eval_results/*/paired_results.csv`.

Usage
-----
  python MujocoDeploy/record_rma_videos.py \
      --ckpt logs/h1_2_rma/Apr16_15-12-01_/model_6000.pt \
      --tag highlight

  # custom conditions
  python MujocoDeploy/record_rma_videos.py --ckpt <ckpt> \
      --conditions torso:100:+Y:walk:0 torso:150:+X:walk:0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np
import torch
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mujoco

from MujocoDeploy.comprehensive_eval import COMMANDS
from MujocoDeploy.sweep_rma_forces import (
    AXIS_DIRECTIONS,
    BODY_KEY_MAP,
    compute_obs,
    load_models,
    normalize_et_np,
    pd_control,
    quat_rotate_inverse,
)


# ─── Defaults: the conditions we identified as most informative ──────────
# Picked from the stress sweep CSV where RMA survived and baseline fell.
DEFAULT_CONDITIONS = [
    # (body, magnitude_N, direction_label, command_tag, seed)
    ("torso",       100, "+Y", "walk", 0),
    ("torso",       150, "+X", "walk", 0),
    ("left_wrist",  100, "+Y", "walk", 1),
    ("right_wrist", 100, "-Y", "walk", 0),
]


# ─── Condition spec ─────────────────────────────────────────────────────
@dataclass
class VideoCondition:
    body: str
    magnitude: float
    direction_tag: str
    command_tag: str
    seed: int
    torso_force: np.ndarray
    left_wrist_force: np.ndarray
    right_wrist_force: np.ndarray

    @property
    def name(self) -> str:
        return f"{self.body}_{int(self.magnitude)}N_{self.direction_tag}_{self.command_tag}_seed{self.seed}"


def parse_condition(s: str) -> VideoCondition:
    """'body:mag:dir:cmd:seed' → VideoCondition."""
    parts = s.split(":")
    if len(parts) != 5:
        raise ValueError(f"Condition must be body:mag:dir:cmd:seed, got {s!r}")
    body, mag_s, dname, cmd_tag, seed_s = parts
    return build_condition(body, float(mag_s), dname, cmd_tag, int(seed_s))


def build_condition(body: str, mag: float, dname: str, cmd_tag: str, seed: int) -> VideoCondition:
    if body not in BODY_KEY_MAP:
        raise ValueError(f"body must be one of {list(BODY_KEY_MAP)}, got {body!r}")
    if dname not in AXIS_DIRECTIONS:
        raise ValueError(f"direction must be one of {list(AXIS_DIRECTIONS)}, got {dname!r}")
    if cmd_tag not in COMMANDS:
        raise ValueError(f"command must be one of {list(COMMANDS)}, got {cmd_tag!r}")

    zero = np.zeros(3, dtype=np.float32)
    force = AXIS_DIRECTIONS[dname].astype(np.float32) * mag
    forces = {k: zero.copy() for k in BODY_KEY_MAP.values()}
    forces[BODY_KEY_MAP[body]] = force
    return VideoCondition(
        body=body, magnitude=mag, direction_tag=dname, command_tag=cmd_tag, seed=seed,
        torso_force=forces["torso_force"],
        left_wrist_force=forces["left_wrist_force"],
        right_wrist_force=forces["right_wrist_force"],
    )


# ─── Recording helpers ───────────────────────────────────────────────────
def draw_overlay(frame: np.ndarray, text_lines: List[str], color=(255, 255, 255),
                 bg_alpha=0.5) -> np.ndarray:
    """Draw a semi-transparent overlay at the top-left of the frame."""
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    line_h = 24
    pad = 8
    width = max(cv2.getTextSize(t, font, scale, thickness)[0][0] for t in text_lines) + 2 * pad
    height = line_h * len(text_lines) + 2 * pad

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, out, 1 - bg_alpha, 0, out)

    y = pad + line_h - 6
    for line in text_lines:
        cv2.putText(out, line, (pad, y), font, scale, color, thickness, cv2.LINE_AA)
        y += line_h
    return out


# ─── Single-trial runner with rendering ─────────────────────────────────
def run_trial_render(m, policy, encoder, cfg, cond: VideoCondition,
                     use_encoder: bool, renderer: "mujoco.Renderer",
                     render_fps=30) -> Tuple[List[np.ndarray], dict]:
    """Run one trial and return (frames, metrics)."""
    eval_cfg = cfg["eval"]
    dt = cfg["simulation_dt"]
    decim = cfg["control_decimation"]
    n_leg = cfg["num_actions"]
    n_steps = int(eval_cfg["duration"] / dt)
    fall_height = eval_cfg["fall_height"]
    force_start = eval_cfg.get("force_start_time", 1.0)
    phase_period = cfg.get("phase_period", 0.8)
    max_tau = 300.0

    # One render frame every N sim steps
    steps_per_frame = max(1, int(round(1.0 / (render_fps * dt))))

    d = mujoco.MjData(m)
    rng = np.random.default_rng(cond.seed)
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

    # Camera that tracks the torso
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = torso_id
    cam.distance = 3.5
    cam.azimuth = 130
    cam.elevation = -15
    cam.lookat[:] = d.qpos[0:3]

    action = np.zeros(n_leg, dtype=np.float32)
    counter = 0
    survival_time = eval_cfg["duration"]
    frames: List[np.ndarray] = []

    cmd = COMMANDS[cond.command_tag].astype(np.float32)

    method_label = "RMA" if use_encoder else "Baseline (no encoder)"

    for step in range(n_steps):
        t = step * dt

        # External force
        d.xfrc_applied[:] = 0
        if t >= force_start:
            if torso_id >= 0:
                d.xfrc_applied[torso_id, :3] = cond.torso_force
            if left_id >= 0:
                d.xfrc_applied[left_id, :3] = cond.left_wrist_force
            if right_id >= 0:
                d.xfrc_applied[right_id, :3] = cond.right_wrist_force

        # PD legs
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

        mujoco.mj_step(m, d)
        counter += 1

        # Fall check
        if d.qpos[2] < fall_height:
            survival_time = t
            # Render a few extra frames post-fall for visual clarity
            for _ in range(int(render_fps * 0.3)):
                mujoco.mj_step(m, d)
                renderer.update_scene(d, cam)
                frame = renderer.render()
                frames.append(_annotate(frame, cond, method_label, t, fallen=True))
            break

        # Policy step
        if counter % decim == 0:
            phase = (t / phase_period) % 1.0
            obs_47, _ = compute_obs(d, cfg, action, cmd, phase, n_leg)

            if use_encoder and t >= force_start:
                e_t = np.concatenate([cond.torso_force, cond.left_wrist_force,
                                      cond.right_wrist_force]).astype(np.float32)
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

        # Render frame
        if counter % steps_per_frame == 0:
            renderer.update_scene(d, cam)
            frame = renderer.render()
            frames.append(_annotate(frame, cond, method_label, t, fallen=False))

    return frames, {
        "survival_time": round(survival_time, 3),
        "success": survival_time >= eval_cfg["duration"] - 0.01,
        "method": method_label,
    }


def _annotate(frame_rgb: np.ndarray, cond: VideoCondition, method_label: str,
              t: float, fallen: bool) -> np.ndarray:
    """Add text overlay to the frame (in-place copy) and return BGR for cv2."""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    target_force = {"torso": cond.torso_force, "left_wrist": cond.left_wrist_force,
                    "right_wrist": cond.right_wrist_force}[cond.body]
    lines = [
        f"{method_label}",
        f"{cond.body}  |  {int(cond.magnitude)}N {cond.direction_tag}",
        f"cmd={cond.command_tag}  seed={cond.seed}",
        f"t={t:5.2f}s" + ("   FALLEN" if fallen else ""),
    ]
    color = (60, 180, 60) if "RMA" in method_label else (60, 60, 220)  # BGR
    return draw_overlay(frame_bgr, lines, color=color)


def write_mp4(frames: List[np.ndarray], path: str, fps: int = 30):
    if not frames:
        print(f"  (no frames for {path})")
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"  wrote {path}  ({len(frames)} frames, {len(frames)/fps:.1f}s)")


def stitch_side_by_side(frames_a: List[np.ndarray], frames_b: List[np.ndarray],
                        label_a="RMA", label_b="Baseline") -> List[np.ndarray]:
    """Pad the shorter list with its last frame; then hstack."""
    n = max(len(frames_a), len(frames_b))

    def pad(fs):
        if not fs:
            return []
        return fs + [fs[-1]] * (n - len(fs))

    a, b = pad(frames_a), pad(frames_b)
    out = []
    for fa, fb in zip(a, b):
        # Guard against different sizes (shouldn't happen but be safe)
        if fa.shape != fb.shape:
            fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]))
        # Divider
        div = np.full((fa.shape[0], 4, 3), 255, dtype=np.uint8)
        out.append(np.hstack([fa, div, fb]))
    return out


# ─── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Record paired RMA vs baseline videos.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Training checkpoint .pt")
    parser.add_argument("--config", type=str,
                        default=os.path.join(_SCRIPT_DIR, "sweep_config.yaml"))
    parser.add_argument("--tag", type=str, default="highlight")
    parser.add_argument("--out_root", type=str,
                        default=os.path.join(_SCRIPT_DIR, "videos"))
    parser.add_argument("--conditions", type=str, nargs="+", default=None,
                        help="List of body:mag:dir:cmd:seed. "
                             "Omit to use the curated default set.")
    parser.add_argument("--duration", type=float, default=8.0,
                        help="Trial duration (s) — keep short to bound file size.")
    parser.add_argument("--force_start", type=float, default=1.0,
                        help="When to apply forces (s).")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=30)
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
    cfg["eval"]["force_start_time"] = args.force_start

    # ---- Conditions ----
    if args.conditions:
        conditions = [parse_condition(s) for s in args.conditions]
    else:
        conditions = [build_condition(*c) for c in DEFAULT_CONDITIONS]

    # ---- Output dir ----
    ckpt_stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir = os.path.join(args.out_root, f"{ckpt_stem}__{args.tag}__{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ---- Load models ----
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.abspath(args.ckpt)
    policy, encoder = load_models(cfg, ckpt_path)
    print(f"Loaded checkpoint: {ckpt_path}")

    # ---- MuJoCo model ----
    m = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    m.opt.timestep = cfg["simulation_dt"]

    # ---- Single renderer reused across trials ----
    renderer = mujoco.Renderer(m, height=args.height, width=args.width)

    # ---- Run each condition twice ----
    summary_lines = ["# Paired video recordings", ""]
    t0 = time.time()
    for i, cond in enumerate(conditions):
        print(f"\n[{i+1}/{len(conditions)}] {cond.name}")
        frames_rma, m_rma = run_trial_render(m, policy, encoder, cfg, cond,
                                             use_encoder=True, renderer=renderer,
                                             render_fps=args.fps)
        frames_base, m_base = run_trial_render(m, policy, encoder, cfg, cond,
                                               use_encoder=False, renderer=renderer,
                                               render_fps=args.fps)

        rma_path = os.path.join(out_dir, f"{cond.name}__RMA.mp4")
        base_path = os.path.join(out_dir, f"{cond.name}__baseline.mp4")
        comp_path = os.path.join(out_dir, f"{cond.name}__compare.mp4")
        write_mp4(frames_rma, rma_path, args.fps)
        write_mp4(frames_base, base_path, args.fps)
        comp = stitch_side_by_side(frames_rma, frames_base)
        write_mp4(comp, comp_path, args.fps)

        summary_lines.append(
            f"- **{cond.name}** — RMA survived {m_rma['survival_time']}s "
            f"(success={m_rma['success']}), baseline {m_base['survival_time']}s "
            f"(success={m_base['success']})"
        )

    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"All videos in: {out_dir}")


if __name__ == "__main__":
    main()
