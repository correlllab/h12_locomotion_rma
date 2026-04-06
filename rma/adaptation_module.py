"""1D CNN adaptation module for RMA Phase 2 (deployment).

Maps history of (obs, action) -> latent z_t.
Same architecture as homie_h12. Created now for completeness; not used in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Adaptation1DCNNCfg:
    in_channels: int           # obs_dim + action_dim per timestep
    history_length: int = 30
    latent_dim: int = 8
    hidden_dims: tuple[int, ...] = (512, 256, 128)


class Adaptation1DCNN(nn.Module):
    """1D CNN adaptation module: history window -> latent z_t."""

    def __init__(self, cfg: Adaptation1DCNNCfg):
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = []
        prev_channels = cfg.in_channels
        for hidden_ch in cfg.hidden_dims:
            layers.append(nn.Conv1d(prev_channels, hidden_ch, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            prev_channels = hidden_ch
        self.conv_net = nn.Sequential(*layers)
        self.fc = nn.Linear(prev_channels * cfg.history_length, cfg.latent_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Args: history (B, history_length * in_channels). Returns: z_t (B, latent_dim)."""
        B = history.shape[0]
        x = history.view(B, self.cfg.history_length, self.cfg.in_channels)
        x = x.transpose(1, 2)  # (B, in_channels, history_length)
        x = self.conv_net(x)
        x = x.reshape(B, -1)
        return self.fc(x)
