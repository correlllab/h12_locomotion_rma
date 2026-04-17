"""Wrapper that keeps the RMA encoder in PPO's computation graph.

Problem it solves:
  In the original runner, the encoder ran inside torch.inference_mode()
  during collection, and PPO stored pre-computed [base_obs | z_t] as fixed
  observations.  During PPO updates the encoder was never in the backward
  graph, so it received NO reinforcement-learning gradient — only a separate
  reconstruction loss trained it, which teaches faithful encoding but not
  *control-useful* encoding.

Solution:
  Store [base_obs | e_t_norm] (raw normalised forces) in PPO storage.
  This wrapper sits between PPO and the real ActorCriticRecurrent.  On every
  forward call it splits the observation, runs the encoder to get z_t, and
  forwards [base_obs | z_t] to the inner network.  During PPO mini-batch
  updates the encoder is in the live computation graph, so RL gradients
  (surrogate + value loss) flow through it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RmaActorCriticWrapper(nn.Module):
    """Thin wrapper: [base_obs | e_t_norm] → encoder → [base_obs | z_t] → actor-critic."""

    is_recurrent = True

    def __init__(
        self,
        actor_critic: nn.Module,
        encoder: nn.Module,
        num_base_actor_obs: int,
        num_base_critic_obs: int,
        et_dim: int,
    ):
        super().__init__()
        self.actor_critic = actor_critic
        self.encoder = encoder
        self._num_base_actor_obs = num_base_actor_obs
        self._num_base_critic_obs = num_base_critic_obs
        self._et_dim = et_dim

    # -------------------------------------------------------------- #
    #  Internal: split → encode → concatenate
    # -------------------------------------------------------------- #
    def _encode_and_augment(self, obs: torch.Tensor, base_dim: int) -> torch.Tensor:
        """[base_obs(base_dim) | e_t_norm(et_dim)] → [base_obs | z_t]."""
        base = obs[..., :base_dim]
        et = obs[..., base_dim: base_dim + self._et_dim]
        z_t = self.encoder(et)
        return torch.cat([base, z_t], dim=-1)

    # -------------------------------------------------------------- #
    #  ActorCriticRecurrent interface (forwarded to inner module)
    # -------------------------------------------------------------- #
    def act(self, observations, masks=None, hidden_states=None):
        aug = self._encode_and_augment(observations, self._num_base_actor_obs)
        return self.actor_critic.act(aug, masks=masks, hidden_states=hidden_states)

    def act_inference(self, observations):
        aug = self._encode_and_augment(observations, self._num_base_actor_obs)
        return self.actor_critic.act_inference(aug)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        aug = self._encode_and_augment(critic_observations, self._num_base_critic_obs)
        return self.actor_critic.evaluate(aug, masks=masks, hidden_states=hidden_states)

    def get_hidden_states(self):
        return self.actor_critic.get_hidden_states()

    def reset(self, dones=None):
        return self.actor_critic.reset(dones)

    def get_actions_log_prob(self, actions):
        return self.actor_critic.get_actions_log_prob(actions)

    # ---- properties that PPO reads directly ----
    @property
    def std(self):
        return self.actor_critic.std

    @property
    def action_mean(self):
        return self.actor_critic.action_mean

    @property
    def action_std(self):
        return self.actor_critic.action_std

    @property
    def entropy(self):
        return self.actor_critic.entropy
