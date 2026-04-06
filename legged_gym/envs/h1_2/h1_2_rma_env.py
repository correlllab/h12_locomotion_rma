"""H1-2 RMA environment with force randomization on torso + wrists.

Key differences from H1_2Robot:
- Uses handless URDF (27 DOFs): 12 leg DOFs policy-actuated, 15 upper body PD-held.
- Applies randomized external forces to torso_link, left/right wrist_roll_link.
- Exposes rma_et (9-dim force vector) for the runner to encode into z_t.
- Observations only include 12 leg DOFs (same obs structure as base H1_2).
- Reward terms identical to unitree rl gym.
"""

import time
import numpy as np
import torch

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor

from rma.gym_et_builder import (
    sample_rma_forces,
    resample_rma_forces_for_envs,
    build_et,
    make_rma_force_tensor,
)
from rma.env_factor_spec import RMA_RESAMPLE_PROB


class H1_2RmaRobot(LeggedRobot):
    """H1-2 robot with RMA force randomization (Phase 1)."""

    # ------------------------------------------------------------------ #
    #  Init
    # ------------------------------------------------------------------ #
    def _init_buffers(self):
        """Override to handle num_dof (27) > num_actions (12).

        Sizes torques/p_gains/d_gains by num_dof so PD control covers all
        joints (legs get policy actions, upper body targets defaults).
        """
        # Acquire GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Wrap tensors
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        # Common buffers
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(
            get_axis_params(-1., self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        # KEY: size torques/gains by num_dof (27), not num_actions (12)
        self.torques = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device
        )
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)

        # Actions are still 12 (legs only)
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs, self.cfg.commands.num_commands,
            dtype=torch.float, device=self.device
        )
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device
        )
        self.feet_air_time = torch.zeros(
            self.num_envs, self.feet_indices.shape[0],
            dtype=torch.float, device=self.device
        )
        self.last_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices),
            dtype=torch.bool, device=self.device
        )
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Joint default positions and PD gains for ALL 27 DOFs
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # Noise scale (computed after obs_buf is available)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        # H1_2 specific: feet state
        self._init_foot()

        # RMA specific: force buffers
        self._init_rma_buffers()

    def _init_foot(self):
        """Initialize foot state tracking from rigid body tensors."""
        self.feet_num = len(self.feet_indices)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _init_rma_buffers(self):
        """Initialize RMA force buffers and find body indices for force application."""
        # Find body indices for force application
        self._rma_torso_body_idx = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], "torso_link"
        )
        self._rma_left_wrist_body_idx = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], "left_wrist_roll_link"
        )
        self._rma_right_wrist_body_idx = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], "right_wrist_roll_link"
        )

        # Force buffers
        self.rma_torso_force = torch.zeros(self.num_envs, 3, device=self.device)
        self.rma_left_force = torch.zeros(self.num_envs, 3, device=self.device)
        self.rma_right_force = torch.zeros(self.num_envs, 3, device=self.device)

        # e_t buffer (filled each step in compute_observations)
        self.rma_et = torch.zeros(self.num_envs, 9, device=self.device)

        # Resample probability
        self._rma_resample_prob = getattr(self.cfg.rma, 'resample_prob', RMA_RESAMPLE_PROB)

        # Force magnitude range from config
        self._rma_force_range = tuple(getattr(self.cfg.rma, 'force_magnitude_range', [0.0, 10.0]))

        # Sample initial forces
        t, l, r = sample_rma_forces(self.num_envs, self.device, self._rma_force_range)
        self.rma_torso_force[:] = t
        self.rma_left_force[:] = l
        self.rma_right_force[:] = r

    # ------------------------------------------------------------------ #
    #  Step
    # ------------------------------------------------------------------ #
    def step(self, actions):
        """Override to apply RMA external forces during physics simulation."""
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # Resample forces probabilistically
        resample_mask = torch.rand(self.num_envs, device=self.device) < self._rma_resample_prob
        resample_ids = resample_mask.nonzero(as_tuple=False).flatten()
        resample_rma_forces_for_envs(
            self.rma_torso_force, self.rma_left_force, self.rma_right_force, resample_ids,
            self._rma_force_range,
        )

        # Build force tensor (applied every substep)
        force_tensor = make_rma_force_tensor(
            self.num_envs, self.num_bodies,
            self._rma_torso_body_idx, self._rma_left_wrist_body_idx, self._rma_right_wrist_body_idx,
            self.rma_torso_force, self.rma_left_force, self.rma_right_force,
            self.device,
        )

        # Physics simulation with external forces
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            # Apply RMA external forces (world frame)
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(force_tensor),
                None,
                gymapi.ENV_SPACE,
            )
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time - elapsed_time > 0:
                    time.sleep(sim_time - elapsed_time)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self.post_physics_step()

        # Clip observations
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ------------------------------------------------------------------ #
    #  Torques: 12 policy actions -> 27 DOF torques
    # ------------------------------------------------------------------ #
    def _compute_torques(self, actions):
        """Compute PD torques for all 27 DOFs.

        Leg DOFs (0-11): target = action * scale + default
        Upper body DOFs (12-26): target = default (PD holding)
        """
        actions_scaled = actions * self.cfg.control.action_scale  # (N, 12)
        # Pad to full DOF space: upper body gets zero action offset
        full_actions_scaled = torch.zeros(
            self.num_envs, self.num_dof, device=self.device
        )
        full_actions_scaled[:, :self.num_actions] = actions_scaled
        torques = (
            self.p_gains * (full_actions_scaled + self.default_dof_pos - self.dof_pos)
            - self.d_gains * self.dof_vel
        )
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # ------------------------------------------------------------------ #
    #  Observations
    # ------------------------------------------------------------------ #
    def _get_noise_scale_vec(self, cfg):
        """Noise scaling for 47-dim obs (same structure as H1_2)."""
        noise_vec = torch.zeros(self.cfg.env.num_observations, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.  # commands
        noise_vec[9:9 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9 + self.num_actions:9 + 2 * self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9 + 2 * self.num_actions:9 + 3 * self.num_actions] = 0.  # previous actions
        noise_vec[9 + 3 * self.num_actions:9 + 3 * self.num_actions + 2] = 0.  # sin/cos phase
        return noise_vec

    def compute_observations(self):
        """Compute 47-dim obs (leg DOFs only) + 50-dim privileged obs + e_t."""
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)

        # Only leg DOFs (first 12) in observations
        leg_dof_pos = self.dof_pos[:, :self.num_actions]
        leg_dof_vel = self.dof_vel[:, :self.num_actions]
        leg_default = self.default_dof_pos[:, :self.num_actions]

        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (leg_dof_pos - leg_default) * self.obs_scales.dof_pos,
            leg_dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phase,
            cos_phase,
        ), dim=-1)

        self.privileged_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (leg_dof_pos - leg_default) * self.obs_scales.dof_pos,
            leg_dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phase,
            cos_phase,
        ), dim=-1)

        # Add noise
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # Build e_t (forces only, 9 dims) for the runner
        self.rma_et = build_et(self.rma_torso_force, self.rma_left_force, self.rma_right_force)

    # ------------------------------------------------------------------ #
    #  Post-physics + callbacks
    # ------------------------------------------------------------------ #
    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _post_physics_step_callback(self):
        self.update_feet_state()
        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat(
            [self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1
        )
        return super()._post_physics_step_callback()

    def reset_idx(self, env_ids):
        """Reset environments and resample forces."""
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            resample_rma_forces_for_envs(
                self.rma_torso_force, self.rma_left_force, self.rma_right_force, env_ids,
                self._rma_force_range,
            )

    # ------------------------------------------------------------------ #
    #  Reward functions (identical to unitree rl gym H1_2)
    # ------------------------------------------------------------------ #
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))

    def _reward_alive(self):
        return 1.0

    def _reward_contact_no_vel(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, [0, 2, 6, 8]]), dim=1)

    # ---- Override base DOF rewards to only consider leg DOFs (first 12) ----
    # With 27 DOFs (12 legs + 15 upper body PD-held), the base class sums
    # over all DOFs, inflating penalties from non-policy-controlled joints.
    def _reward_torques(self):
        return torch.sum(torch.square(self.torques[:, :self.num_actions]), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel[:, :self.num_actions]), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square(
            (self.last_dof_vel[:, :self.num_actions] - self.dof_vel[:, :self.num_actions]) / self.dt
        ), dim=1)
