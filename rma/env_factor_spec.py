"""Environment factor (e_t) specification for H12 RMA — paper-full variant.

Extends the original forces-only e_t with the privileged dynamics factors
randomized in the RMA paper (Kumar et al. 2021, Table I):

  e_t = [
      torso_force        (3)   N      # retained from base branch
      left_wrist_force   (3)   N      # retained from base branch
      right_wrist_force  (3)   N      # retained from base branch
      base_mass_offset   (1)   kg     # paper: "Payload"
      com_offset         (3)   m      # paper: "Center of Mass"
      motor_strength     (12)  -      # paper: per-joint torque scale
      friction           (1)   -      # paper: ground friction μ
  ]  →  26 dims total.

Each group has its own normalization range, so the encoder sees every
component in [-1, 1] regardless of physical units.
"""

from __future__ import annotations

from dataclasses import dataclass


# ─── Group ranges ─────────────────────────────────────────────────────
# Forces: spherical sampling; magnitude in [0, max]. Component range sets
# the normalization bounds (±max_force maps to ±1 on each axis).
FORCE_MAGNITUDE_RANGE: tuple[float, float] = (0.0, 100.0)          # N
FORCE_COMPONENT_RANGE: tuple[float, float] = (-100.0, 100.0)       # N

# Paper Table I values (adapted from A1 quadruped → H1-2 humanoid;
# see note on COM below — the paper reports ±0.15 cm which is tiny
# even for A1, we use ±0.05 m as a humanoid-scale reinterpretation).
BASE_MASS_RANGE: tuple[float, float] = (0.0, 6.0)                  # kg  (paper: [0, 6])
COM_OFFSET_RANGE: tuple[float, float] = (-0.05, 0.05)              # m
MOTOR_STRENGTH_RANGE: tuple[float, float] = (0.90, 1.10)           # scale (paper: [0.90, 1.10])
FRICTION_RANGE: tuple[float, float] = (0.1, 2.0)                   # μ  (IsaacGym/PhysX humanoid; paper [0.05, 4.5] is RaiSim/A1)

# Per-step resample probability for the time-varying component (forces).
# Paper uses 0.004 for all factors; we keep the existing 0.01 for forces
# but leave static factors fixed per episode (sampled at reset).
RMA_RESAMPLE_PROB: float = 0.01


# ─── Spec & slicing ──────────────────────────────────────────────────
@dataclass(frozen=True)
class RmaEtSpec:
    """Layout of the 26-dim e_t vector."""

    torso_force_dim: int = 3
    hand_force_dim: int = 3
    num_hands: int = 2
    base_mass_dim: int = 1
    com_offset_dim: int = 3
    motor_strength_dim: int = 12
    friction_dim: int = 1

    @property
    def dim(self) -> int:
        return (
            self.torso_force_dim
            + self.num_hands * self.hand_force_dim
            + self.base_mass_dim
            + self.com_offset_dim
            + self.motor_strength_dim
            + self.friction_dim
        )

    # Offsets into the flat e_t vector
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

    @property
    def force_slice(self) -> slice:
        """All nine force components together (for legacy tooling)."""
        return slice(0, self.torso_force_dim + self.num_hands * self.hand_force_dim)

    @property
    def base_mass_slice(self) -> slice:
        start = self.torso_force_dim + self.num_hands * self.hand_force_dim
        return slice(start, start + self.base_mass_dim)

    @property
    def com_offset_slice(self) -> slice:
        start = (self.torso_force_dim + self.num_hands * self.hand_force_dim
                 + self.base_mass_dim)
        return slice(start, start + self.com_offset_dim)

    @property
    def motor_strength_slice(self) -> slice:
        start = (self.torso_force_dim + self.num_hands * self.hand_force_dim
                 + self.base_mass_dim + self.com_offset_dim)
        return slice(start, start + self.motor_strength_dim)

    @property
    def friction_slice(self) -> slice:
        start = (self.torso_force_dim + self.num_hands * self.hand_force_dim
                 + self.base_mass_dim + self.com_offset_dim
                 + self.motor_strength_dim)
        return slice(start, start + self.friction_dim)


DEFAULT_ET_SPEC = RmaEtSpec()


# ─── Normalization ────────────────────────────────────────────────────
def _norm_group(x, lo: float, hi: float):
    """Map [lo, hi] → [-1, 1] element-wise (supports numpy/torch via ops)."""
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def _denorm_group(x_norm, lo: float, hi: float):
    return (x_norm + 1.0) * (hi - lo) / 2.0 + lo


def normalize_et(et, spec: RmaEtSpec = DEFAULT_ET_SPEC):
    """Normalize each group of e_t to [-1, 1] using its physical range.

    Works element-wise with a torch.Tensor or a numpy array; the .clone()
    below guards against in-place mutation of the raw e_t buffer.
    """
    out = et.clone() if hasattr(et, "clone") else et.copy()
    fr_lo, fr_hi = FORCE_COMPONENT_RANGE
    out[..., spec.force_slice] = _norm_group(et[..., spec.force_slice], fr_lo, fr_hi)
    out[..., spec.base_mass_slice] = _norm_group(
        et[..., spec.base_mass_slice], *BASE_MASS_RANGE
    )
    out[..., spec.com_offset_slice] = _norm_group(
        et[..., spec.com_offset_slice], *COM_OFFSET_RANGE
    )
    out[..., spec.motor_strength_slice] = _norm_group(
        et[..., spec.motor_strength_slice], *MOTOR_STRENGTH_RANGE
    )
    out[..., spec.friction_slice] = _norm_group(
        et[..., spec.friction_slice], *FRICTION_RANGE
    )
    return out


def denormalize_et(et_norm, spec: RmaEtSpec = DEFAULT_ET_SPEC):
    """Inverse of normalize_et."""
    out = et_norm.clone() if hasattr(et_norm, "clone") else et_norm.copy()
    fr_lo, fr_hi = FORCE_COMPONENT_RANGE
    out[..., spec.force_slice] = _denorm_group(et_norm[..., spec.force_slice], fr_lo, fr_hi)
    out[..., spec.base_mass_slice] = _denorm_group(
        et_norm[..., spec.base_mass_slice], *BASE_MASS_RANGE
    )
    out[..., spec.com_offset_slice] = _denorm_group(
        et_norm[..., spec.com_offset_slice], *COM_OFFSET_RANGE
    )
    out[..., spec.motor_strength_slice] = _denorm_group(
        et_norm[..., spec.motor_strength_slice], *MOTOR_STRENGTH_RANGE
    )
    out[..., spec.friction_slice] = _denorm_group(
        et_norm[..., spec.friction_slice], *FRICTION_RANGE
    )
    return out


# ─── Force-application body names ────────────────────────────────────
RMA_FORCE_BODY_NAMES: tuple[str, ...] = (
    "torso_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
)
