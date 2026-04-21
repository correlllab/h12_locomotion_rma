"""1D CNN adaptation module for RMA Phase 2.

Architecture follows Kumar et al. 2021 (RMA paper, Section IV-B):
  1. Per-timestep 2-layer MLP embed: in_channels -> 32
  2. Three 1D-CNN layers across the time axis:
        (32, 32, k=8, s=4)
        (32, 32, k=5, s=1)
        (32, 32, k=5, s=1)
  3. Flatten + linear -> latent z_t.

For history_length=50 (paper k=50) the temporal length evolves
50 -> 11 -> 7 -> 3, giving a flattened 96-dim vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Adaptation1DCNNCfg:
    in_channels: int           # obs_dim + action_dim per timestep
    history_length: int = 50
    latent_dim: int = 8
    embed_dim: int = 32


class Adaptation1DCNN(nn.Module):
    """History window (B, H*C) -> latent z_t."""

    def __init__(self, cfg: Adaptation1DCNNCfg):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Sequential(
            nn.Linear(cfg.in_channels, cfg.embed_dim),
            nn.ReLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.ReLU(),
        )

        self.conv_net = nn.Sequential(
            nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size=5, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, cfg.embed_dim, cfg.history_length)
            flat_dim = self.conv_net(dummy).reshape(1, -1).shape[1]
        self.fc = nn.Linear(flat_dim, cfg.latent_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: (B, history_length * in_channels) -> (B, latent_dim)."""
        B = history.shape[0]
        x = history.view(B, self.cfg.history_length, self.cfg.in_channels)
        x = self.embed(x)                    # (B, H, embed_dim)
        x = x.transpose(1, 2)                # (B, embed_dim, H)
        x = self.conv_net(x)                 # (B, embed_dim, H')
        x = x.reshape(B, -1)
        return self.fc(x)
