from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class SahrRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 1024
        num_actions = 10
        num_observations = 229
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_yaw': 0., #0.79*np.pi/180,   # [rad]
            'left_hip_roll': 0., #6.06*np.pi/180,   # [rad]
            'left_hip_pitch': 0., #-31.99*np.pi/180,  # [rad]
            'left_knee': 0., #-42.18*np.pi/180,  # [rad]
            'left_ankle_pitch': 0., #-21.53*np.pi/180,     # [rad]
            
            'right_hip_yaw': 0., #0.79*np.pi/180,   # [rad]
            'right_hip_roll': 0., #6.06*np.pi/180,     # [rad]
            'right_hip_pitch': 0., #-31.99*np.pi/180 ,  # [rad]
            'right_knee': 0., #-42.18*np.pi/180,  # [rad]
            'right_ankle_pitch': 0., #-21.53*np.pi/180,

            'head_yaw': 0.0,
            'head_pitch': 0.0,
            
            'right_shoulder_pitch': 0.0,
            'right_shoulder_roll': 0.0,
            'right_elbow': 0.0,

            'left_shoulder_pitch': 0.0,
            'left_shoulder_roll': 0.0,
            'left_elbow': 0.0,
        }


    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = { 'left_hip_yaw': 20.,
            'left_hip_roll': 20., 'left_hip_pitch': 20., 'left_knee': 20., 
            'left_ankle_pitch': 20., 'right_hip_yaw': 20.,
            'right_hip_roll': 20., 'right_hip_pitch': 20., 'right_knee': 20.,
            'right_ankle_pitch': 20. }
        damping = { 'left_knee': 0.5, 'left_hip_yaw': 0.5,
            'left_hip_roll': 0.5, 'left_hip_pitch': 0.5, 
            'left_ankle_pitch': 0.5, 'right_knee': 0.5, 'right_hip_yaw': 0.5,
            'right_hip_roll': 0.5, 'right_hip_pitch': 0.5, 
            'right_ankle_pitch': 0.5 }   # [N*m*s/rad]
        
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/sahr/mjcf/sahr.xml'
        name = "sahr"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["trunk_1", "u_shoulder_1", "u_shoulder_2",
                                       "right_humerus_1", "left_humerus_1",
                                       "left_boxing_glove",
                                       "right_knee_1", "left_knee_1"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

class SahrRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_sahr'