from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class AlienGoCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        #num_observations = 48
        num_observations = 45
        episode_length_s = 20 
  
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False # if you change this you have to change _get_noise_scale_vector
  
    class asset( LeggedRobotCfg.asset ):
        name = "aliengo"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo_description/urdf/aliengo.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["trunk"]
        terminate_after_contacts_on = ["trunk", "base", "hip", "thigh"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        # fix_base_link = True

    
    class commands( LeggedRobotCfg.commands ):
        #heading_command = False
        resampling_time = 4.
        class ranges( LeggedRobotCfg.commands.ranges ):
            ang_vel_yaw = [-1.5, 1.5]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.50] # x,y,z [m]
        default_joint_angles = {
             'FL_hip_joint': 0.0,   # [rad]v
             'RL_hip_joint': 0.0,   # [rad]v
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
        only_positive_rewards = True
        class scales( LeggedRobotCfg.rewards.scales ):
            box_topple = 0.
            dist_from_box = 2.

            termination = -0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time = 0.0
            collision = -0.0
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.0

    class noise( LeggedRobotCfg.noise ):
        add_noise = False
    
    class domain_rand( LeggedRobotCfg.domain_rand ):
        friction_range = [0., 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.

class AlienGoRoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'aliengo'
        max_iterations = 3000 # number of policy updates
        
    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01