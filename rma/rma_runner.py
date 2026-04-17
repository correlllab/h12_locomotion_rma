"""RMA Phase 1 on-policy runner — end-to-end encoder training.

Key design:
  PPO stores [base_obs | e_t_norm] (47+9=56 actor, 50+9=59 critic).
  An RmaActorCriticWrapper sits between PPO and the real ActorCriticRecurrent.
  On every forward call the wrapper encodes e_t → z_t and forwards
  [base_obs | z_t] to the inner network.  During PPO mini-batch updates the
  encoder is live in the computation graph, so RL gradients (surrogate +
  value loss) flow through it — the encoder learns to produce z_t values
  that are *useful for control*, not merely reconstructable.

  A separate decoder is still trained with MSE reconstruction loss as a
  diagnostic / regulariser, but does NOT back-prop into the encoder (decoder
  only).
"""

import os
import time
import statistics
from collections import deque

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv

from rma.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg
from rma.env_factor_decoder import EnvFactorDecoder, EnvFactorDecoderCfg
from rma.env_factor_spec import normalize_et
from rma.rma_actor_critic_wrapper import RmaActorCriticWrapper


class RmaOnPolicyRunner:
    """On-policy runner with RMA Phase 1 (end-to-end encoder via wrapper)."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device='cpu'):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.rma_cfg = train_cfg.get("rma", {})
        self.device = device
        self.env = env

        # ---- RMA parameters ----
        self.rma_latent_dim = self.rma_cfg.get("latent_dim", 8)
        self.rma_et_dim = self.rma_cfg.get("et_dim", 9)
        self.rma_recon_coef = self.rma_cfg.get("recon_coef", 0.5)

        # ---- Inner actor-critic dimensions (what the LSTM actually sees) ----
        num_inner_actor_obs = self.env.num_obs + self.rma_latent_dim   # 47 + 8 = 55
        if self.env.num_privileged_obs is not None:
            num_inner_critic_obs = self.env.num_privileged_obs + self.rma_latent_dim  # 50 + 8 = 58
        else:
            num_inner_critic_obs = num_inner_actor_obs

        # ---- Storage dimensions (what PPO stores: base_obs + e_t_norm) ----
        num_storage_actor_obs = self.env.num_obs + self.rma_et_dim    # 47 + 9 = 56
        if self.env.num_privileged_obs is not None:
            num_storage_critic_obs = self.env.num_privileged_obs + self.rma_et_dim  # 50 + 9 = 59
        else:
            num_storage_critic_obs = num_storage_actor_obs

        # ---- Create inner actor-critic (receives [base_obs | z_t]) ----
        actor_critic_class = eval(self.cfg["policy_class_name"])
        inner_actor_critic: ActorCritic = actor_critic_class(
            num_inner_actor_obs, num_inner_critic_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        # ---- Create encoder ----
        self.encoder = EnvFactorEncoder(EnvFactorEncoderCfg(
            in_dim=self.rma_et_dim,
            latent_dim=self.rma_latent_dim,
            hidden_dims=tuple(self.rma_cfg.get("encoder_hidden_dims", [256, 128])),
        )).to(self.device)

        # ---- Create wrapper (encoder + actor-critic, end-to-end) ----
        #   PPO sees the wrapper as its actor_critic.  The wrapper's
        #   parameters() includes both the inner actor-critic AND the encoder,
        #   so PPO's Adam optimizer trains both with RL gradients.
        wrapper = RmaActorCriticWrapper(
            actor_critic=inner_actor_critic,
            encoder=self.encoder,
            num_base_actor_obs=self.env.num_obs,
            num_base_critic_obs=(self.env.num_privileged_obs
                                if self.env.num_privileged_obs is not None
                                else self.env.num_obs),
            et_dim=self.rma_et_dim,
        ).to(self.device)

        # ---- Create decoder (reconstruction diagnostic, trained separately) ----
        self.decoder = EnvFactorDecoder(EnvFactorDecoderCfg(
            in_dim=self.rma_latent_dim,
            out_dim=self.rma_et_dim,
            hidden_dims=tuple(self.rma_cfg.get("decoder_hidden_dims", [256, 128])),
            use_output_scaling=False,
        )).to(self.device)

        # ---- Create PPO with the *wrapper* as actor_critic ----
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO = alg_class(wrapper, device=self.device, **self.alg_cfg)

        # ---- Replace PPO optimizer with param-group version ----
        #   Encoder gets a smaller RL learning rate to prevent z_t from
        #   shifting too fast between PPO mini-batches (which would violate
        #   the trust region and destabilise training).
        enc_lr_scale = self.rma_cfg.get("encoder_rl_lr_scale", 0.1)
        base_lr = self.alg.learning_rate
        encoder_param_ids = set(id(p) for p in self.encoder.parameters())
        ac_params = [p for p in wrapper.parameters()
                     if id(p) not in encoder_param_ids]
        self.alg.optimizer = torch.optim.Adam([
            {"params": ac_params, "lr": base_lr},
            {"params": list(self.encoder.parameters()),
             "lr": base_lr * enc_lr_scale},
        ])

        # ---- Decoder-only optimizer (reconstruction loss, no encoder) ----
        self.rma_optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr=self.rma_cfg.get("encoder_lr", 1e-3),
        )

        # ---- Force curriculum ----
        self._curriculum_steps = self.rma_cfg.get("curriculum_steps", 0)
        self._max_force = self.rma_cfg.get("max_force",
                                           self.env._rma_force_range[1]
                                           if hasattr(self.env, '_rma_force_range')
                                           else 100.0)

        # ---- Init storage with *storage* obs shapes (base + e_t_norm) ----
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.alg.init_storage(
            self.env.num_envs, self.num_steps_per_env,
            [num_storage_actor_obs], [num_storage_critic_obs],
            [self.env.num_actions],
        )

        # ---- RMA e_t buffer (for reconstruction loss) ----
        self.rma_et_buffer = torch.zeros(
            self.num_steps_per_env, self.env.num_envs, self.rma_et_dim,
            device=self.device,
        )

        # ---- Logging ----
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    # -------------------------------------------------------------- #
    #  Convenience: unwrap inner actor-critic
    # -------------------------------------------------------------- #
    @property
    def _inner_actor_critic(self):
        """The real ActorCriticRecurrent inside the wrapper."""
        return self.alg.actor_critic.actor_critic

    # -------------------------------------------------------------- #
    #  Force curriculum
    # -------------------------------------------------------------- #
    def _update_force_curriculum(self, iteration: int) -> float:
        """Linearly ramp force magnitude from 0 to max over curriculum_steps.

        Returns the current max force for logging.
        """
        if self._curriculum_steps <= 0 or not hasattr(self.env, '_rma_force_range'):
            return self._max_force

        progress = min(1.0, iteration / self._curriculum_steps)
        current_max = self._max_force * progress
        self.env._rma_force_range = (0.0, current_max)
        return current_max

    # ------------------------------------------------------------------ #
    #  Training loop
    # ------------------------------------------------------------------ #
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()   # wrapper (includes encoder + inner AC)
        self.decoder.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # ---- Force curriculum: ramp force range over training ----
            current_max_force = self._update_force_curriculum(it)

            # ---- Rollout collection ----
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # Get e_t from env and normalize to [-1, 1]
                    e_t_raw = self.env.rma_et                  # (N, 9)
                    e_t_norm = normalize_et(e_t_raw)           # (N, 9)

                    # Store [base_obs | e_t_norm] — the wrapper will encode
                    # e_t → z_t during its forward pass (both here in
                    # inference_mode and later during PPO update with grads).
                    augmented_obs = torch.cat([obs, e_t_norm], dim=-1)           # (N, 56)
                    augmented_critic = torch.cat([critic_obs, e_t_norm], dim=-1) # (N, 59)

                    # Store e_t for decoder reconstruction loss
                    self.rma_et_buffer[i] = e_t_norm

                    # PPO act (wrapper encodes e_t → z_t internally)
                    actions = self.alg.act(augmented_obs, augmented_critic)

                    # Env step
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Compute returns (wrapper encodes e_t internally)
                start = stop
                e_t_raw = self.env.rma_et
                e_t_norm = normalize_et(e_t_raw)
                last_augmented_critic = torch.cat([critic_obs, e_t_norm], dim=-1)
                self.alg.compute_returns(last_augmented_critic)

            # ---- PPO update (RL gradients flow through wrapper → encoder) ----
            mean_value_loss, mean_surrogate_loss = self.alg.update()

            # Restore encoder LR ratio — PPO's adaptive KL schedule sets
            # all param groups to the same LR, overriding our scaling.
            # Param group 1 is the encoder (see __init__).
            if len(self.alg.optimizer.param_groups) > 1:
                enc_lr_scale = self.rma_cfg.get("encoder_rl_lr_scale", 0.1)
                self.alg.optimizer.param_groups[1]['lr'] = (
                    self.alg.learning_rate * enc_lr_scale
                )

            # ---- Decoder reconstruction update (decoder only) ----
            mean_recon_loss = self._rma_update()

            stop = time.time()
            learn_time = stop - start

            # ---- Logging ----
            if self.log_dir is not None:
                self._log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(
            self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)
        ))

    # ------------------------------------------------------------------ #
    #  Decoder reconstruction update (decoder only, encoder gets RL grads)
    # ------------------------------------------------------------------ #
    def _rma_update(self) -> float:
        """Train decoder with MSE reconstruction loss.

        The encoder is in the forward graph (z = encoder(et)), but
        rma_optimizer only manages decoder parameters, so only the decoder
        is updated here.  We explicitly zero stale encoder gradients
        afterward to keep things clean for the next PPO update cycle.
        """
        # Flatten: (T, N, 9) -> (T*N, 9)
        et_flat = self.rma_et_buffer.flatten(0, 1)

        # Forward — encoder is in graph but only decoder is optimised
        z = self.encoder(et_flat)
        recon_loss = self.decoder.compute_reconstruction_loss(z, et_flat)
        loss = self.rma_recon_coef * recon_loss

        # Backward
        self.rma_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
        self.rma_optimizer.step()

        # Zero stale encoder grads from recon loss (PPO optimizer manages
        # encoder grads; we don't want leftover recon grads leaking in).
        self.encoder.zero_grad()

        return recon_loss.item()

    # ------------------------------------------------------------------ #
    #  Logging
    # ------------------------------------------------------------------ #
    def _log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = ''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/rma_reconstruction', locs['mean_recon_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        self.writer.add_scalar('RMA/max_force_curriculum', locs['current_max_force'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        if len(locs['rewbuffer']) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f""" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m \n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'RMA reconstruction loss:':>{pad}} {locs['mean_recon_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f""" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m \n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'RMA reconstruction loss:':>{pad}} {locs['mean_recon_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    # ------------------------------------------------------------------ #
    #  Save / Load
    # ------------------------------------------------------------------ #
    def save(self, path, infos=None):
        """Save checkpoint.

        Stores inner actor-critic and encoder state dicts separately
        (same format as before) so deployment scripts work unchanged.
        """
        torch.save({
            'model_state_dict': self._inner_actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'rma_optimizer_state_dict': self.rma_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self._inner_actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if 'encoder_state_dict' in loaded_dict:
            self.encoder.load_state_dict(loaded_dict['encoder_state_dict'])
        if 'decoder_state_dict' in loaded_dict:
            self.decoder.load_state_dict(loaded_dict['decoder_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            if 'rma_optimizer_state_dict' in loaded_dict:
                self.rma_optimizer.load_state_dict(loaded_dict['rma_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        """Return inference function compatible with play.py.

        Returns a closure that automatically fetches e_t from the env,
        normalises it, concatenates with base obs, and runs the wrapper's
        act_inference (which encodes e_t → z_t internally).  This way
        callers (like play.py) can pass plain 47-dim obs and it just works.
        """
        self.alg.actor_critic.eval()  # wrapper (encoder + inner AC)
        if device is not None:
            self.alg.actor_critic.to(device)

        wrapper = self.alg.actor_critic
        env = self.env

        def _inference(obs):
            e_t_norm = normalize_et(env.rma_et).to(obs.device)
            augmented = torch.cat([obs, e_t_norm], dim=-1)
            return wrapper.act_inference(augmented)

        return _inference
