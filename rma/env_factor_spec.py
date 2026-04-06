"""Environment factor (e_t) specification for H12 RMA (forces-only, with torso).

e_t = torso_force(3) + left_wrist_force(3) + right_wrist_force(3) = 9 dims.
Forces are sampled in 3D via spherical sampling (magnitude then direction)
and applied to torso_link, left_wrist_roll_link, right_wrist_roll_link in world frame.
"""

from __future__ import annotations

from dataclasses import dataclass


# Force: spherical sampling. Magnitude in [0, max] (N); direction uniform on unit sphere.
FORCE_MAGNITUDE_RANGE: tuple[float, float] = (0.0, 10.0)  # N
# Per-axis range for decoder output scaling
FORCE_COMPONENT_RANGE: tuple[float, float] = (-10.0, 10.0)

# Per-step resample probability (RMA paper default: 0.004).
RMA_RESAMPLE_PROB: float = 0.004


@dataclass(frozen=True)
class RmaEtSpec:
    """e_t for H12 RMA (forces-only): torso(3) + left_wrist(3) + right_wrist(3) = 9."""

    torso_force_dim: int = 3
    hand_force_dim: int = 3
    num_hands: int = 2

    @property
    def dim(self) -> int:
        return self.torso_force_dim + self.num_hands * self.hand_force_dim

    @property
    def torso_force_slice(self) -> slice:
        return slice(0, self.torso_force_dim)

    @property
    def left_wrist_force_slice(self) -> slice:
        start = self.torso_force_dim
        return slice(start, start + self.hand_force_dim)

    @property
    def right_wrist_force_slice(self) -> slice:
        start = self.torso_force_dim + self.hand_force_dim
        return slice(start, start + self.hand_force_dim)


DEFAULT_ET_SPEC = RmaEtSpec()

# Body links where forces are applied
RMA_FORCE_BODY_NAMES: tuple[str, ...] = (
    "torso_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
)
