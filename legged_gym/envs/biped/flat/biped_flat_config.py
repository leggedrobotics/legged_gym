from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class BipedFlatCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 42
        num_actions = 10

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class commands(LeggedRobotCfg.commands):
        heading_command = False

    class ranges(LeggedRobotCfg.commands.ranges):
        lin_vel_x = [-0.5, 0.5] # min max [m/s]
        lin_vel_y = [0.0, 0.0]   # min max [m/s]
        ang_vel_yaw = [0, 0]    # min max [rad/s]
        heading = [0, 0]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'L_hip_joint': 0,
            'L_hip2_joint': 0,
            'R_hip_joint': 0,
            'R_hip2_joint': 0,
            
            'L_thigh_joint': 0,
            'R_thigh_joint': 0,
            
            'L_calf_joint': 0,
            'R_calf_joint': 0,
            
            'L_toe_joint': 0,
            'R_toe_joint': 0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {   'hip_joint': 30.0, 'hip2_joint': 30.0,
                        'thigh_joint': 30., 'calf_joint': 30.,
                        'toe_joint': 30.}  # [N*m/rad]
        damping = { 'hip_joint': 0.5, 'hip2_joint': 0.5,
                    'thigh_joint': 0.5, 'calf_joint': 0.5,
                    'toe_joint': 0.5}  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/biped/biped_simple_osudrl.xml'
        name = "biped"
        foot_name = 'toe'
        terminate_after_contacts_on = ['pelvis']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            tracking_ang_vel = 1.0
            torques = -5.e-6
            dof_acc = -2.e-7
            lin_vel_z = -0.5
            feet_air_time = 5.
            dof_pos_limits = -1.
            no_fly = 0.25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.

class BipedFlatCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'biped_flat'

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01