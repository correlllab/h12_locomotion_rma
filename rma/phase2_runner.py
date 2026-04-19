"""RMA Phase 2 trainer — distill teacher's z_t into a history-based student.

Phase 1 trained a teacher policy π(a | obs, z_t) + encoder μ(e_t → z_t) jointly
via PPO.  At deployment we don't have access to e_t (privileged env factors),
so Phase 2 learns an adaptation module φ(h_t → ẑ_t) from the history of
proprioceptive (obs, action) pairs.  φ is trained with MSE supervised regression
against the frozen teacher's z_t.

Rollout strategy: on-policy with the TEACHER.  The env steps under the teacher's
actions, the teacher's encoder produces ground-truth z_t from e_t, and φ
predicts ẑ_t from the history we've been accumulating.  This is the vanilla
RMA recipe (no DAgger switching in the paper).

At deployment we can either swap in φ (z_t → ẑ_t from history) and keep the
teacher's inner actor-critic, or fine-tune end-to-end — we just do the former.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from rma.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg
from rma.env_factor_spec import normalize_et
from rma.adaptation_module import Adaptation1DCNN, Adaptation1DCNNCfg


@dataclass
class Phase2Cfg:
    history_length: int = 30
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    lr: float = 5e-4
    grad_clip: float = 1.0
    num_steps_per_update: int = 24        # same as PPO's num_steps_per_env for similar data volume
    num_iterations: int = 5000
    save_interval: int = 100
    log_interval: int = 10


class RmaPhase2Runner:
    """Distil teacher encoder → history-based student.

    Args:
        env: a ready H1_2RmaRobot instance (paper-full, exposes `rma_et`).
        teacher_ckpt: path to Phase 1 `.pt` with encoder_state_dict.
        rma_cfg: dict with `latent_dim`, `et_dim`, `encoder_hidden_dims`.
        cfg: Phase2Cfg with trainer hyperparameters.
        log_dir: where to dump tensorboard events + checkpoints.
        device: torch device.
    """

    def __init__(
        self,
        env,
        teacher_encoder: nn.Module,
        teacher_wrapper: nn.Module,
        rma_cfg: dict,
        cfg: Phase2Cfg,
        log_dir: str | None,
        device: str = "cuda:0",
    ):
        self.env = env
        self.cfg = cfg
        self.device = device
        self.log_dir = log_dir

        self.et_dim = int(rma_cfg.get("et_dim", 26))
        self.latent_dim = int(rma_cfg.get("latent_dim", 8))

        # ---- Teacher encoder + wrapper (frozen; wrapper includes encoder + LSTM) ----
        self.encoder = teacher_encoder
        self.teacher = teacher_wrapper
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        # Encoder is inside wrapper but we use it directly for ground-truth z_t
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # ---- Student adaptation module ----
        self.in_channels = self.env.num_obs + self.env.num_actions  # e.g. 47 + 12 = 59
        self.adaptation = Adaptation1DCNN(Adaptation1DCNNCfg(
            in_channels=self.in_channels,
            history_length=cfg.history_length,
            latent_dim=self.latent_dim,
            hidden_dims=cfg.hidden_dims,
        )).to(device)

        self.optim = torch.optim.Adam(self.adaptation.parameters(), lr=cfg.lr)

        # ---- History buffer (num_envs, H, in_channels) — rolling per env ----
        self.history = torch.zeros(
            self.env.num_envs,
            cfg.history_length,
            self.in_channels,
            device=device,
        )

        self.writer: SummaryWriter | None = None
        self.total_steps = 0
        self.current_iter = 0

    # -------------------------------------------------------------- #
    #  Teacher rollout (act + recurrent-state reset)
    # -------------------------------------------------------------- #
    def _teacher_act(self, obs: torch.Tensor) -> torch.Tensor:
        """Run the frozen teacher: encode env.rma_et → z_t, LSTM → action."""
        e_t_norm = normalize_et(self.env.rma_et).to(obs.device)
        augmented = torch.cat([obs, e_t_norm], dim=-1)
        return self.teacher.act_inference(augmented)

    def _reset_teacher_state(self, dones: torch.Tensor) -> None:
        """Reset LSTM hidden state for terminated envs."""
        if dones.any():
            self.teacher.reset(dones)

    # -------------------------------------------------------------- #
    #  History buffer maintenance
    # -------------------------------------------------------------- #
    def _push_history(self, obs: torch.Tensor, actions: torch.Tensor):
        """Append (obs, action) to the rolling window; oldest step falls off."""
        new_step = torch.cat([obs, actions], dim=-1).unsqueeze(1)  # (N, 1, in_ch)
        self.history = torch.cat([self.history[:, 1:], new_step], dim=1)

    def _zero_history_for_dones(self, dones: torch.Tensor):
        """When an env resets, its history is stale — zero it out."""
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            self.history[done_ids] = 0.0

    # -------------------------------------------------------------- #
    #  Training loop
    # -------------------------------------------------------------- #
    def learn(self, num_iterations: int | None = None):
        if num_iterations is None:
            num_iterations = self.cfg.num_iterations
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        obs = self.env.get_observations().to(self.device)
        tot_iter = self.current_iter + num_iterations
        loss_buf: deque[float] = deque(maxlen=100)
        mae_buf: deque[float] = deque(maxlen=100)

        for it in range(self.current_iter, tot_iter):
            start = time.time()

            batch_preds: list[torch.Tensor] = []
            batch_targets: list[torch.Tensor] = []

            for _ in range(self.cfg.num_steps_per_update):
                # (1) predict ẑ_t from history (gradients enabled)
                pred_z = self.adaptation(
                    self.history.reshape(self.env.num_envs, -1)
                )  # (N, latent_dim)

                # (2) ground-truth z_t from teacher encoder (no grad)
                with torch.no_grad():
                    e_t_norm = normalize_et(self.env.rma_et).to(self.device)
                    teacher_z = self.encoder(e_t_norm)

                batch_preds.append(pred_z)
                batch_targets.append(teacher_z.detach())

                # (3) step the env under TEACHER policy (no grad; env is stateful)
                with torch.inference_mode():
                    actions = self._teacher_act(obs)
                new_obs, _, _, dones, _ = self.env.step(actions)
                dones = dones.to(self.device)

                # (4) update history with (obs, actions) BEFORE overwriting obs
                self._push_history(obs, actions)
                self._zero_history_for_dones(dones)
                self._reset_teacher_state(dones)
                obs = new_obs.to(self.device)
                self.total_steps += self.env.num_envs

            # (5) supervised MSE update over the collected mini-batch
            preds = torch.cat(batch_preds, dim=0)
            targets = torch.cat(batch_targets, dim=0)
            loss = torch.nn.functional.mse_loss(preds, targets)

            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.adaptation.parameters(), self.cfg.grad_clip)
            self.optim.step()

            with torch.no_grad():
                mae = (preds - targets).abs().mean().item()
            loss_buf.append(loss.item())
            mae_buf.append(mae)

            elapsed = time.time() - start

            if self.writer is not None:
                self.writer.add_scalar("Phase2/mse_loss", loss.item(), it)
                self.writer.add_scalar("Phase2/mae", mae, it)
                self.writer.add_scalar("Phase2/steps_per_sec",
                                       self.env.num_envs * self.cfg.num_steps_per_update / elapsed,
                                       it)

            if it % self.cfg.log_interval == 0:
                print(
                    f"[iter {it:>5}/{tot_iter}] "
                    f"loss={sum(loss_buf)/len(loss_buf):.5f} "
                    f"mae={sum(mae_buf)/len(mae_buf):.5f} "
                    f"time={elapsed:.2f}s  total_steps={self.total_steps}"
                )

            if self.log_dir is not None and it % self.cfg.save_interval == 0 and it > 0:
                self.save(os.path.join(self.log_dir, f"adaptation_{it}.pt"))

            self.current_iter = it + 1

        # Final save
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"adaptation_{self.current_iter}.pt"))

    # -------------------------------------------------------------- #
    #  Save / load
    # -------------------------------------------------------------- #
    def save(self, path: str):
        torch.save({
            "adaptation_state_dict": self.adaptation.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "iter": self.current_iter,
            "cfg": {
                "in_channels": self.in_channels,
                "history_length": self.cfg.history_length,
                "latent_dim": self.latent_dim,
                "hidden_dims": list(self.cfg.hidden_dims),
            },
        }, path)
        print(f"[Phase2] saved checkpoint → {path}")

    def load(self, path: str, load_optimizer: bool = True):
        sd = torch.load(path, map_location=self.device)
        self.adaptation.load_state_dict(sd["adaptation_state_dict"])
        if load_optimizer and "optimizer_state_dict" in sd:
            self.optim.load_state_dict(sd["optimizer_state_dict"])
        self.current_iter = sd.get("iter", 0)
        print(f"[Phase2] loaded adaptation module from {path} (iter={self.current_iter})")
