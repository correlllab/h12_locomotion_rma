"""RMA (Rapid Motor Adaptation) modules for H1-2 locomotion.

Phase 1 (training, sim):
  - Sample forces: sample_rma_forces() for torso + both wrists.
  - Build e_t: build_et(torso, left, right) -> 9-dim force vector.
  - Apply forces: make_rma_force_tensor() -> gym.apply_rigid_body_force_tensors().
  - Resample with prob RMA_RESAMPLE_PROB (0.01) via resample_rma_forces_for_envs().
  - Encoder: e_t (9) -> z_t (8); policy(obs, z_t). Decoder for reconstruction loss.

Phase 2 (deploy):
  - Adaptation1DCNN(history of obs, action) -> z_hat; policy(obs, z_hat).
"""

from .env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg
from .env_factor_decoder import EnvFactorDecoder, EnvFactorDecoderCfg
from .adaptation_module import Adaptation1DCNN, Adaptation1DCNNCfg
from .rma_actor_critic_wrapper import RmaActorCriticWrapper
from .env_factor_spec import (
    DEFAULT_ET_SPEC,
    FORCE_COMPONENT_RANGE,
    FORCE_MAGNITUDE_RANGE,
    RMA_FORCE_BODY_NAMES,
    RMA_RESAMPLE_PROB,
    RmaEtSpec,
    normalize_et,
    denormalize_et,
)
from .gym_et_builder import (
    build_et,
    make_rma_force_tensor,
    resample_rma_forces_for_envs,
    sample_rma_forces,
)

__all__ = [
    "EnvFactorEncoder",
    "EnvFactorEncoderCfg",
    "EnvFactorDecoder",
    "EnvFactorDecoderCfg",
    "RmaActorCriticWrapper",
    "Adaptation1DCNN",
    "Adaptation1DCNNCfg",
    "RmaEtSpec",
    "DEFAULT_ET_SPEC",
    "RMA_FORCE_BODY_NAMES",
    "FORCE_MAGNITUDE_RANGE",
    "FORCE_COMPONENT_RANGE",
    "RMA_RESAMPLE_PROB",
    "normalize_et",
    "denormalize_et",
    "build_et",
    "sample_rma_forces",
    "resample_rma_forces_for_envs",
    "make_rma_force_tensor",
]
