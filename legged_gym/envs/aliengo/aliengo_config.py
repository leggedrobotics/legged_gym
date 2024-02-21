from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class AlienGoCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_observations = 48
  
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
  
    class asset( LeggedRobotCfg.asset ):
        name = "aliengo"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo_description/urdf/aliengo.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["trunk"]
        terminate_after_contacts_on = ["trunk", "base", "hip", "thigh"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        # fix_base_link = True

    class commands( LeggedRobotCfg.commands ):
        heading_command = False
        resampling_time = 4.
        class ranges( LeggedRobotCfg.commands.ranges ):
            ang_vel_yaw = [-1.5, 1.5]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.50] # x,y,z [m]
        default_joint_angles = {
             'FL_hip_joint': 0.0,   # [rad]
             'RL_hip_joint': 0.0,   # [rad]
             'FR_hip_joint': -0.0 ,  # [rad]
             'RR_hip_joint': -0.0,   # [rad]

             'FL_thigh_joint': .8,     # [rad]
             'RL_thigh_joint': 1.2,   # [rad]
             'FR_thigh_joint': 0.8,     # [rad]
             'RR_thigh_joint': 1.2,   # [rad]

             'FL_calf_joint': -1.25,   # [rad]
             'RL_calf_joint': -1.25,    # [rad]
             'FR_calf_joint': -1.25,  # [rad]
             'RR_calf_joint': -1.25,    # [rad]
        }
    
    class control ( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'

        stiffness = {'joint': 60.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 1.
        max_contact_force = 300.
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            pass
    
    class domain_rand( LeggedRobotCfg.domain_rand ):
        friction_range = [0., 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.

class AlienGoRoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'aliengo'
        
    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01