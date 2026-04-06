from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class H1_2RmaRoughCfg(LeggedRobotCfg):
    """Environment config for H1-2 RMA locomotion.

    Uses handless URDF (27 DOFs) with only 12 leg DOFs actuated by policy.
    Upper body (15 DOFs) held at default positions with PD gains.
    External forces randomized on torso + both wrists for RMA.
    Reward terms identical to unitree rl gym H1_2.
    """

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.05]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # --- Legs (12 DOF, policy-actuated) ---
            'left_hip_yaw_joint': 0,
            'left_hip_pitch_joint': -0.16,
            'left_hip_roll_joint': 0,
            'left_knee_joint': 0.36,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            'right_hip_yaw_joint': 0,
            'right_hip_pitch_joint': -0.16,
            'right_hip_roll_joint': 0,
            'right_knee_joint': 0.36,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
            # --- Upper body (15 DOF, PD-held at defaults) ---
            'torso_joint': 0,
            'left_shoulder_pitch_joint': 0.4,
            'left_shoulder_roll_joint': 0,
            'left_shoulder_yaw_joint': 0,
            'left_elbow_joint': 0.3,
            'left_wrist_roll_joint': 0,
            'left_wrist_pitch_joint': 0,
            'left_wrist_yaw_joint': 0,
            'right_shoulder_pitch_joint': 0.4,
            'right_shoulder_roll_joint': 0,
            'right_shoulder_yaw_joint': 0,
            'right_elbow_joint': 0.3,
            'right_wrist_roll_joint': 0,
            'right_wrist_pitch_joint': 0,
            'right_wrist_yaw_joint': 0,
        }

    class env(LeggedRobotCfg.env):
        # 3 + 3 + 3 + 12 + 12 + 12 + 2 = 47 (same as unitree rl gym H1_2)
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12  # Only leg DOFs controlled by policy

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {
            # Legs (policy-controlled)
            'hip_yaw_joint': 200.,
            'hip_roll_joint': 200.,
            'hip_pitch_joint': 200.,
            'knee_joint': 300.,
            'ankle_pitch_joint': 40.,
            'ankle_roll_joint': 40.,
            # Upper body (PD holding gains)
            'torso_joint': 200.,
            'shoulder_pitch_joint': 100.,
            'shoulder_roll_joint': 100.,
            'shoulder_yaw_joint': 100.,
            'elbow_joint': 100.,
            'wrist_roll_joint': 50.,
            'wrist_pitch_joint': 50.,
            'wrist_yaw_joint': 50.,
        }  # [N*m/rad]
        damping = {
            # Legs
            'hip_yaw_joint': 2.5,
            'hip_roll_joint': 2.5,
            'hip_pitch_joint': 2.5,
            'knee_joint': 4,
            'ankle_pitch_joint': 2.0,
            'ankle_roll_joint': 2.0,
            # Upper body
            'torso_joint': 5.0,
            'shoulder_pitch_joint': 3.0,
            'shoulder_roll_joint': 3.0,
            'shoulder_yaw_joint': 3.0,
            'elbow_joint': 3.0,
            'wrist_roll_joint': 2.0,
            'wrist_pitch_joint': 2.0,
            'wrist_yaw_joint': 2.0,
        }  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 8

    class sim(LeggedRobotCfg.sim):
        dt = 0.0025

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/h1_2_handless_homie.urdf'
        name = "h1_2"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0
        flip_visual_attachments = False
        armature = 1e-3

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 1.0
        only_positive_rewards = False  # Need negative signal for value function with force perturbations

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

    class rma:
        """RMA force randomization parameters (used by env)."""
        resample_prob = 0.004
        force_magnitude_range = [0.0, 10.0]


class H1_2RmaRoughCfgPPO(LeggedRobotCfgPPO):
    """Training config for H1-2 RMA Phase 1."""
    runner_class_name = 'RmaOnPolicyRunner'

    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu'
        # ActorCriticRecurrent (LSTM):
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        algorithm_class_name = 'PPO'
        max_iterations = 10000
        run_name = ''
        experiment_name = 'h1_2_rma'

    class rma:
        """RMA encoder/decoder architecture and training params."""
        et_dim = 9          # e_t: torso(3) + left_wrist(3) + right_wrist(3)
        latent_dim = 8      # z_t dimension
        encoder_hidden_dims = [256, 128]
        decoder_hidden_dims = [256, 128]
        recon_coef = 0.5    # Reconstruction loss coefficient
        encoder_lr = 1e-3   # Separate LR for encoder/decoder
