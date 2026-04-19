"""Phase 2 RMA trainer entry point.

Loads a Phase 1 teacher (actor-critic + encoder), freezes it, and trains the
1D-CNN adaptation module to regress z_t from a short history of
(obs, action).

Example:
    python legged_gym/scripts/train_phase2.py \
        --task h1_2_rma --headless --num_envs 1024 \
        --teacher_run Apr19_12-40-02_paperfull_v1 --teacher_ckpt 900 \
        --phase2_iterations 5000 --run_name phase2_v1
"""

import argparse
import os
from datetime import datetime

import isaacgym  # noqa: F401 must import before torch
import torch

from legged_gym.envs import *  # registers tasks
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path, class_to_dict

from rma.phase2_runner import RmaPhase2Runner, Phase2Cfg


def _add_phase2_args():
    """Return argparse-compatible entries for gymutil's custom_parameters.

    We reuse get_args() for consistency with train.py, but extend it with
    Phase 2-specific options via monkeypatch on gymutil.
    """
    # Done inline in main() via env-var escape hatch — see below.


def parse_phase2_args():
    # We reuse get_args() so the user can pass the standard IsaacGym args
    # (--task, --headless, --num_envs, --rl_device, ...), but add a few
    # Phase 2 specific ones via a second parser that consumes the remaining.
    args = get_args()
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_run", type=str, default=None,
                   help="Phase 1 run directory (e.g. Apr19_12-40-02_paperfull_v1); "
                        "default = latest run in logs/<experiment>/")
    p.add_argument("--teacher_ckpt", type=int, default=-1,
                   help="Phase 1 checkpoint iteration (default: latest)")
    p.add_argument("--phase2_iterations", type=int, default=5000,
                   help="Number of Phase 2 training iterations")
    p.add_argument("--phase2_lr", type=float, default=5e-4)
    p.add_argument("--phase2_hist_len", type=int, default=30)
    p.add_argument("--phase2_run_name", type=str, default="phase2")
    phase2_args, _ = p.parse_known_args()
    for k, v in vars(phase2_args).items():
        setattr(args, k, v)
    return args


def main():
    args = parse_phase2_args()

    # ---- Build env (paper-full, exposes rma_et) ----
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    # ---- Build Phase 1 runner so we get the teacher actor-critic loaded ----
    # We pass args.resume=True + load_run/checkpoint so make_alg_runner
    # populates the teacher weights.  We don't call .learn() on it.
    args.resume = True
    if args.teacher_run is not None:
        args.load_run = args.teacher_run
    if args.teacher_ckpt is not None and args.teacher_ckpt != -1:
        args.checkpoint = args.teacher_ckpt
    teacher_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args,
    )

    # `train_cfg` is a class-based config; `rma` is a nested class.
    rma_dict = class_to_dict(train_cfg.rma) if hasattr(train_cfg, "rma") else {}

    # ---- Phase 2 log dir ----
    experiment_name = train_cfg.runner.experiment_name  # e.g. h1_2_rma
    log_root = os.path.join("logs", f"{experiment_name}_phase2")
    run_dir_name = f"{datetime.now().strftime('%b%d_%H-%M-%S')}_{args.phase2_run_name}"
    log_dir = os.path.join(log_root, run_dir_name)
    os.makedirs(log_dir, exist_ok=True)

    # ---- Phase 2 runner ----
    cfg = Phase2Cfg(
        history_length=args.phase2_hist_len,
        num_iterations=args.phase2_iterations,
        lr=args.phase2_lr,
    )
    # Put teacher into eval mode (no grads, no dropout, ...)
    teacher_runner.alg.actor_critic.eval()
    teacher_runner.encoder.eval()
    p2 = RmaPhase2Runner(
        env=env,
        teacher_encoder=teacher_runner.encoder,
        teacher_wrapper=teacher_runner.alg.actor_critic,
        rma_cfg=rma_dict,
        cfg=cfg,
        log_dir=log_dir,
        device=args.rl_device,
    )

    print(f"[Phase2] log_dir = {log_dir}")
    print(f"[Phase2] in_channels = {p2.in_channels}, "
          f"history_length = {cfg.history_length}, "
          f"latent_dim = {p2.latent_dim}, et_dim = {p2.et_dim}")

    p2.learn(num_iterations=cfg.num_iterations)


if __name__ == "__main__":
    main()
