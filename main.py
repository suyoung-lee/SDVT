#218 v11
"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from the respective config file.
"""
import argparse
import warnings

import numpy as np
import torch
import json

# get configs
from config.gridworld import \
    args_grid_belief_oracle, args_grid_rl2, args_grid_varibad
from config.pointrobot import \
    args_pointrobot_multitask, args_pointrobot_varibad, args_pointrobot_rl2, args_pointrobot_humplik
from config.mujoco import \
    args_cheetah_dir_multitask, args_cheetah_dir_expert, args_cheetah_dir_rl2, args_cheetah_dir_varibad, \
    args_cheetah_vel_multitask, args_cheetah_vel_expert, args_cheetah_vel_rl2, args_cheetah_vel_varibad, \
    args_cheetah_vel_avg, \
    args_ant_dir_multitask, args_ant_dir_expert, args_ant_dir_rl2, args_ant_dir_varibad, \
    args_ant_goal_multitask, args_ant_goal_expert, args_ant_goal_rl2, args_ant_goal_varibad, \
    args_ant_goal_humplik, \
    args_walker_multitask, args_walker_expert, args_walker_avg, args_walker_rl2, args_walker_varibad, \
    args_humanoid_dir_varibad, args_humanoid_dir_rl2, args_humanoid_dir_multitask, args_humanoid_dir_expert
from config.ml10 import \
    args_ml10_varibad
from environments.parallel_envs import make_vec_envs
from learner import Learner
from metalearner import MetaLearner
from metalearner_ml10 import MetaLearnerML10
from metaeval_ml10 import MetaEvalML10
from metalearner_ml10_post import MetaLearnerML10Post
from metalearner_ml10_post2 import MetaLearnerML10Post2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='gridworld_varibad')
    parser.add_argument('--load-dir', default=None)
    parser.add_argument('--load-iter', default=None)
    args, rest_args = parser.parse_known_args()
    env = args.env_type

    # --- GridWorld ---

    if env == 'gridworld_belief_oracle':
        args = args_grid_belief_oracle.get_args(rest_args)
    elif env == 'gridworld_varibad':
        args = args_grid_varibad.get_args(rest_args)
    elif env == 'gridworld_rl2':
        args = args_grid_rl2.get_args(rest_args)

    # --- PointRobot 2D Navigation ---

    elif env == 'pointrobot_multitask':
        args = args_pointrobot_multitask.get_args(rest_args)
    elif env == 'pointrobot_varibad':
        args = args_pointrobot_varibad.get_args(rest_args)
    elif env == 'pointrobot_rl2':
        args = args_pointrobot_rl2.get_args(rest_args)
    elif env == 'pointrobot_humplik':
        args = args_pointrobot_humplik.get_args(rest_args)

    # --- MUJOCO ---

    # - CheetahDir -
    elif env == 'cheetah_dir_multitask':
        args = args_cheetah_dir_multitask.get_args(rest_args)
    elif env == 'cheetah_dir_expert':
        args = args_cheetah_dir_expert.get_args(rest_args)
    elif env == 'cheetah_dir_varibad':
        args = args_cheetah_dir_varibad.get_args(rest_args)
    elif env == 'cheetah_dir_rl2':
        args = args_cheetah_dir_rl2.get_args(rest_args)
    #
    # - CheetahVel -
    elif env == 'cheetah_vel_multitask':
        args = args_cheetah_vel_multitask.get_args(rest_args)
    elif env == 'cheetah_vel_expert':
        args = args_cheetah_vel_expert.get_args(rest_args)
    elif env == 'cheetah_vel_avg':
        args = args_cheetah_vel_avg.get_args(rest_args)
    elif env == 'cheetah_vel_varibad':
        args = args_cheetah_vel_varibad.get_args(rest_args)
    elif env == 'cheetah_vel_rl2':
        args = args_cheetah_vel_rl2.get_args(rest_args)
    #
    # - AntDir -
    elif env == 'ant_dir_multitask':
        args = args_ant_dir_multitask.get_args(rest_args)
    elif env == 'ant_dir_expert':
        args = args_ant_dir_expert.get_args(rest_args)
    elif env == 'ant_dir_varibad':
        args = args_ant_dir_varibad.get_args(rest_args)
    elif env == 'ant_dir_rl2':
        args = args_ant_dir_rl2.get_args(rest_args)
    #
    # - AntGoal -
    elif env == 'ant_goal_multitask':
        args = args_ant_goal_multitask.get_args(rest_args)
    elif env == 'ant_goal_expert':
        args = args_ant_goal_expert.get_args(rest_args)
    elif env == 'ant_goal_varibad':
        args = args_ant_goal_varibad.get_args(rest_args)
    elif env == 'ant_goal_humplik':
        args = args_ant_goal_humplik.get_args(rest_args)
    elif env == 'ant_goal_rl2':
        args = args_ant_goal_rl2.get_args(rest_args)
    #
    # - Walker -
    elif env == 'walker_multitask':
        args = args_walker_multitask.get_args(rest_args)
    elif env == 'walker_expert':
        args = args_walker_expert.get_args(rest_args)
    elif env == 'walker_avg':
        args = args_walker_avg.get_args(rest_args)
    elif env == 'walker_varibad':
        args = args_walker_varibad.get_args(rest_args)
    elif env == 'walker_rl2':
        args = args_walker_rl2.get_args(rest_args)
    #
    # - HumanoidDir -
    elif env == 'humanoid_dir_multitask':
        args = args_humanoid_dir_multitask.get_args(rest_args)
    elif env == 'humanoid_dir_expert':
        args = args_humanoid_dir_expert.get_args(rest_args)
    elif env == 'humanoid_dir_varibad':
        args = args_humanoid_dir_varibad.get_args(rest_args)
    elif env == 'humanoid_dir_rl2':
        args = args_humanoid_dir_rl2.get_args(rest_args)

    # ml10
    elif env in ['ml10', 'ml10-post', 'ml10-post2']:
        if args.load_dir is None:
            args = args_ml10_varibad.get_args(rest_args)
            args.load_dir = None
            args.load_iter = None
        else:
            load_dir = args.load_dir
            load_iter = args.load_iter
            with open(load_dir + 'config.json', 'r') as f:
                args.__dict__ = json.load(f)
            args.load_dir = load_dir
            args.load_iter = load_iter
            print(args)
    elif env in ['ml10-eval']:
        load_dir = args.load_dir
        load_iter = args.load_iter
        render = args.render
        with open(load_dir + 'config.json', 'r') as f:
            args.__dict__ = json.load(f)
        args.load_dir = load_dir
        args.load_iter = load_iter
        args.render = render
        if render:
            args.env_name = 'ML10RENDEREnv-v2'
        print(args)
        #args = args_ml10_varibad.get_args(rest_args)
    else:
        raise Exception("Invalid Environment")

    # warning for deterministic execution
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    # if we're normalising the actions, we have to make sure that the env expects actions within [-1, 1]
    if args.norm_actions_pre_sampling or args.norm_actions_post_sampling:
        envs = make_vec_envs(env_name=args.env_name, seed=0, num_processes=args.num_processes,
                             gamma=args.policy_gamma, device='cpu',
                             episodes_per_task=args.max_rollouts_per_task,
                             normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                             tasks=None,
                             )
        assert np.unique(envs.action_space.low) == [-1]
        assert np.unique(envs.action_space.high) == [1]

    # clean up arguments
    if args.disable_metalearner or args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False

    if hasattr(args, 'decode_only_past') and args.decode_only_past:
        args.split_batches_by_elbo = True
    # if hasattr(args, 'vae_subsample_decodes') and args.vae_subsample_decodes:
    #     args.split_batches_by_elbo = True

    # begin training (loop through all passed seeds)
    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seed_list:
        print('training', seed)
        args.seed = seed
        args.action_space = None

        if env == 'ml10':
            learner = MetaLearnerML10(args)
        elif env == 'ml10-eval':
            learner = MetaEvalML10(args)
        elif env == 'ml10-post':
            args.results_log_dir = args.results_log_dir + '_post'
            learner = MetaLearnerML10Post(args)
        elif env == 'ml10-post2':
            args.results_log_dir = args.results_log_dir + '_post2'
            learner = MetaLearnerML10Post2(args)
        elif args.disable_metalearner:
            # If `disable_metalearner` is true, the file `learner.py` will be used instead of `metalearner.py`.
            # This is a stripped down version without encoder, decoder, stochastic latent variables, etc.
            learner = Learner(args)
        else:
            learner = MetaLearner(args)
        learner.train()


if __name__ == '__main__':
    main()
