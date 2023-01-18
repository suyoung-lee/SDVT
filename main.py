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
from config.ml10 import \
    args_ml10_SDVT, args_ml10_SD, args_ml10_LDM, args_ml10_VariBAD
from environments.parallel_envs import make_vec_envs
from learner import Learner
from metalearner import MetaLearner
from metalearner_ml10_VariBAD import MetaLearnerML10VariBAD
from metalearner_ml10_SDVT import MetaLearnerML10SDVT
from metalearner_ml10_LDM import MetaLearnerML10LDM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='gridworld_varibad')
    parser.add_argument('--load-dir', default=None)
    parser.add_argument('--load-iter', default=None)
    args, rest_args = parser.parse_known_args()
    env = args.env_type

    # ml10
    if env in ['ml10-SDVT','ml10-SD','ml10-LDM', 'ml10-VariBAD' ]:
        if args.load_dir is None:
            if env == 'ml10-SDVT':
                args = args_ml10_SDVT.get_args(rest_args)
            elif env == 'ml10-SD':
                args = args_ml10_SD.get_args(rest_args)
            elif env == 'ml10-LDM':
                args = args_ml10_LDM.get_args(rest_args)
            elif env == 'ml10-VariBAD':
                args = args_ml10_VariBAD.get_args(rest_args)
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

        if env == 'ml10-SDVT':
            args.results_log_dir = args.results_log_dir
            learner = MetaLearnerML10SDVT(args)
        elif env == 'ml10-SD':
            args.results_log_dir = args.results_log_dir
            learner = MetaLearnerML10SDVT(args)
        elif env == 'ml10-VariBAD':
            args.results_log_dir = args.results_log_dir
            learner = MetaLearnerML10VariBAD(args)
        elif env == 'ml10-LDM':
            args.results_log_dir = args.results_log_dir
            learner = MetaLearnerML10LDM(args)
        elif args.disable_metalearner:
            # If `disable_metalearner` is true, the file `learner.py` will be used instead of `metalearner.py`.
            # This is a stripped down version without encoder, decoder, stochastic latent variables, etc.
            learner = Learner(args)
        else:
            learner = MetaLearner(args)
        learner.train()


if __name__ == '__main__':
    main()
