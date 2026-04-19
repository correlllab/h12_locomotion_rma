"""Build RMA e_t (paper-full, 26 dims) and apply sampled forces.

e_t = torso_force(3) + left(3) + right(3)
      + base_mass(1) + com_offset(3) + motor_strength(12) + friction(1)

Force sampling: magnitude U(0, max), direction uniform on S². Static
dynamics factors (mass, COM, motor strength, friction) are sampled at
episode reset by the env and passed in here.
"""

from __future__ import annotations

import torch

from .env_factor_spec import FORCE_MAGNITUDE_RANGE, DEFAULT_ET_SPEC, RmaEtSpec


def _sample_direction_spherical(n: int, device: torch.device) -> torch.Tensor:
    d = torch.randn(n, 3, device=device, dtype=torch.float32)
    return d / torch.norm(d, dim=1, keepdim=True).clamp(min=1e-6)


def sample_rma_forces(
    num_envs: int,
    device: torch.device,
    force_magnitude_range: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lo, hi = force_magnitude_range if force_magnitude_range is not None else FORCE_MAGNITUDE_RANGE
    forces = []
    for _ in range(3):
        mag = torch.rand(num_envs, 1, device=device, dtype=torch.float32) * (hi - lo) + lo
        forces.append(mag * _sample_direction_spherical(num_envs, device))
    return forces[0], forces[1], forces[2]


def resample_rma_forces_for_envs(
    torso_force: torch.Tensor,
    left_hand_force: torch.Tensor,
    right_hand_force: torch.Tensor,
    env_ids: torch.Tensor,
    force_magnitude_range: tuple[float, float] | None = None,
) -> None:
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
    base_mass_offset: torch.Tensor | None = None,
    com_offset: torch.Tensor | None = None,
    motor_strength: torch.Tensor | None = None,
    friction: torch.Tensor | None = None,
    spec: RmaEtSpec = DEFAULT_ET_SPEC,
) -> torch.Tensor:
    """Concatenate all factor groups into e_t (shape (N, spec.dim)).

    Any group left as None is filled with zeros — useful for
    backwards-compatible construction (forces-only).
    """
    n = torso_force.shape[0]
    device = torso_force.device

    def _z1():
        return torch.zeros(n, 1, device=device)

    def _z3():
        return torch.zeros(n, 3, device=device)

    def _z12():
        return torch.zeros(n, 12, device=device)

    mass = base_mass_offset if base_mass_offset is not None else _z1()
    com = com_offset if com_offset is not None else _z3()
    strength = motor_strength if motor_strength is not None else _z12()
    fric = friction if friction is not None else _z1()

    # Shape-check to fail loudly if a caller passes a wrong-rank tensor
    if mass.dim() == 1:
        mass = mass.unsqueeze(-1)
    if fric.dim() == 1:
        fric = fric.unsqueeze(-1)

    return torch.cat([
        torso_force, left_hand_force, right_hand_force,
        mass, com, strength, fric,
    ], dim=-1)


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
    forces = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float32)
    forces[:, torso_body_index, :] = torso_force
    forces[:, left_wrist_body_index, :] = left_hand_force
    forces[:, right_wrist_body_index, :] = right_hand_force
    return forces
