"""Build RMA e_t and apply sampled forces (torso + wrists).

e_t = torso_force(3) + left_wrist_force(3) + right_wrist_force(3) = 9.
Spherical sampling: magnitude U(0, max), direction = normalize(randn(3)).
"""

from __future__ import annotations

import torch

from .env_factor_spec import FORCE_MAGNITUDE_RANGE


def _sample_direction_spherical(n: int, device: torch.device) -> torch.Tensor:
    """Unit-norm direction: Gaussian then normalize (uniform on sphere). Shape (n, 3)."""
    d = torch.randn(n, 3, device=device, dtype=torch.float32)
    return d / torch.norm(d, dim=1, keepdim=True).clamp(min=1e-6)


def sample_rma_forces(
    num_envs: int,
    device: torch.device,
    force_magnitude_range: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample torso + hand forces via spherical sampling.

    Returns:
        torso_force: (num_envs, 3) in world frame (N)
        left_hand_force: (num_envs, 3) in world frame (N)
        right_hand_force: (num_envs, 3) in world frame (N)
    """
    lo, hi = force_magnitude_range if force_magnitude_range is not None else FORCE_MAGNITUDE_RANGE
    forces = []
    for _ in range(3):  # torso, left, right
        mag = torch.rand(num_envs, 1, device=device, dtype=torch.float32) * (hi - lo) + lo
        direction = _sample_direction_spherical(num_envs, device)
        forces.append(mag * direction)
    return forces[0], forces[1], forces[2]


def resample_rma_forces_for_envs(
    torso_force: torch.Tensor,
    left_hand_force: torch.Tensor,
    right_hand_force: torch.Tensor,
    env_ids: torch.Tensor,
    force_magnitude_range: tuple[float, float] | None = None,
) -> None:
    """In-place resample forces for the given env indices."""
    if env_ids.numel() == 0:
        return
    n = env_ids.shape[0]
    device = torso_force.device
    t, l, r = sample_rma_forces(n, device, force_magnitude_range)
    torso_force[env_ids] = t
    left_hand_force[env_ids] = l
    right_hand_force[env_ids] = r


def build_et(
    torso_force: torch.Tensor,
    left_hand_force: torch.Tensor,
    right_hand_force: torch.Tensor,
) -> torch.Tensor:
    """Build e_t: torso(3) + left(3) + right(3) = 9."""
    return torch.cat([torso_force, left_hand_force, right_hand_force], dim=-1)


def make_rma_force_tensor(
    num_envs: int,
    num_bodies: int,
    torso_body_index: int,
    left_wrist_body_index: int,
    right_wrist_body_index: int,
    torso_force: torch.Tensor,
    left_hand_force: torch.Tensor,
    right_hand_force: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Fill (num_envs, num_bodies, 3) force tensor for gym.apply_rigid_body_force_tensors.

    Tensor is in ENV_SPACE (world frame).
    """
    forces = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float32)
    forces[:, torso_body_index, :] = torso_force
    forces[:, left_wrist_body_index, :] = left_hand_force
    forces[:, right_wrist_body_index, :] = right_hand_force
    return forces
