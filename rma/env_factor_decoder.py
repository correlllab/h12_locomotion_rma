"""Environment factor decoder: latent z_t (8) -> reconstructed e_t (9).

Same MLP architecture as homie_h12, adapted for 9-dim force-only e_t.
Trained with MSE reconstruction loss during RMA Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int, activation: str = "elu") -> nn.Sequential:
    if activation == "elu":
        act: type[nn.Module] = nn.ELU
    elif activation == "relu":
        act = nn.ReLU
    elif activation == "tanh":
        act = nn.Tanh
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    layers: list[nn.Module] = []
    last_dim = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(act())
        last_dim = h
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


@dataclass
class EnvFactorDecoderCfg:
    in_dim: int = 8
    out_dim: int = 9    # torso(3) + left_wrist(3) + right_wrist(3)
    hidden_dims: tuple[int, ...] = (256, 128)
    activation: str = "elu"
    use_output_scaling: bool = False   # Disabled: encoder/decoder work in normalized [-1,1] space
    force_component_range: tuple[float, float] = (-100.0, 100.0)


class EnvFactorDecoder(nn.Module):
    """Decodes latent z_t back to environment factors e_t."""

    def __init__(self, cfg: EnvFactorDecoderCfg | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else EnvFactorDecoderCfg()
        self.net = _build_mlp(
            in_dim=self.cfg.in_dim,
            hidden_dims=list(self.cfg.hidden_dims),
            out_dim=self.cfg.out_dim,
            activation=self.cfg.activation,
        )
        if self.cfg.use_output_scaling:
            lo, hi = self.cfg.force_component_range
            ranges = torch.tensor([[lo, hi]] * self.cfg.out_dim, dtype=torch.float32)
            self.register_buffer("_output_ranges", ranges)

    def forward(self, latent: torch.Tensor, apply_scaling: bool = True) -> torch.Tensor:
        """Decode latent z_t to reconstructed e_t."""
        e_t_raw = self.net(latent)
        if apply_scaling and self.cfg.use_output_scaling:
            ranges = self._output_ranges.to(e_t_raw.device)
            e_t_norm = torch.sigmoid(e_t_raw)
            e_t_raw = ranges[:, 0] + e_t_norm * (ranges[:, 1] - ranges[:, 0])
        return e_t_raw

    def compute_reconstruction_loss(
        self,
        latent: torch.Tensor,
        e_t_target: torch.Tensor,
        apply_scaling: bool = True,
    ) -> torch.Tensor:
        """MSE reconstruction loss: ||decoder(z_t) - e_t||^2."""
        e_t_pred = self.forward(latent, apply_scaling=apply_scaling)
        return nn.functional.mse_loss(e_t_pred, e_t_target)
