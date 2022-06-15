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
    env_cfg.env.num_envs =  min(env_cfg.env.num_envs, 10)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    for i in range(int(10*env.max_episode_length)):   
        act1 = np.ones((env.num_envs, 1))
        act2 = np.zeros((env.num_envs, env.num_actions - 1))
        actions = np.hstack((act1, act2))
        actions = torch.tensor(actions)
        obs, _, rew, done, info = env.step(actions)
    print("Done")

if __name__ == '__main__':
    args = get_args()
    test_env(args)