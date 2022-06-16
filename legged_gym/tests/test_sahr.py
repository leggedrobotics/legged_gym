import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import torch


def test_env(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    for i in range(int(10*env.max_episode_length)):   
        
        actions = torch.tensor([[0.014, 0.106, -0.5583, 0.7362, -0.3758, 0.014, 0.106, -0.5583, 0.7362, -0.3758]])
        obs, _, rew, done, info = env.step(actions)
    print("Done")

if __name__ == '__main__':
    args = get_args()
    test_env(args)