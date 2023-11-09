#- metaeval
import os
import time

import gym
import numpy as np
import torch

from algorithms.a2c import A2C
from algorithms.online_storage import OnlineStorage
from algorithms.ppo import PPO
from environments.parallel_envs import make_vec_envs
from models.policy import Policy
from models.policy_resample import PolicyResample
from utils import evaluation as utl_eval
from utils import helpers as utl
from utils.tb_logger import TBLogger
from vae import VaribadVAE
from vae_mixture import VaribadVAEMixture
from vae_mixture_ext import VaribadVAEMixtureExt
import torch.nn.functional as F

import metaworld
import random
import csv

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MetaEvalML10:
    """
    Meta-Learner class with the main training loop for variBAD.
    """

    def __init__(self, args):

        self.args = args
        if self.args.vae_mixture_num<2:
            self.args.pass_prob_to_policy = False
        utl.seed(self.args.seed, self.args.deterministic_execution)

        # calculate number of updates and keep count of frames/iterations
        self.num_updates = int(args.num_frames) // args.policy_num_steps // args.num_processes
        self.frames = 0
        self.iter_idx = -1
        self.return_list = torch.zeros((self.args.num_processes)).to(device)

        # initialise tensorboard logger
        self.logger = TBLogger(self.args, self.args.exp_label)

        header = ['iter', 'frames']
        for record_type in ['R', 'S', 'SF', 'RF']:
            for task_num in range(15):
                header += [record_type + str(task_num)]
        with open(self.logger.full_output_folder + '/log_eval.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

        self.train_tasks = None

        # initialise environments
        self.envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                                  gamma=args.policy_gamma, device=device,
                                  episodes_per_task=self.args.max_rollouts_per_task,
                                  normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                                  tasks=None
                                  )

        # calculate what the maximum length of the trajectories is
        self.args.max_trajectory_len = self.envs._max_episode_steps
        self.args.max_trajectory_len *= self.args.max_rollouts_per_task

        # get policy input dimensions
        self.args.state_dim = self.envs.observation_space.shape[0]
        self.args.task_dim = self.envs.task_dim

        self.args.belief_dim = self.envs.belief_dim
        self.args.num_states = self.envs.num_states
        # get policy output (action) dimensions
        self.args.action_space = self.envs.action_space

        if isinstance(self.envs.action_space, gym.spaces.discrete.Discrete):
            self.args.action_dim = 1
        else:
            self.args.action_dim = self.envs.action_space.shape[0]

        # initialise VAE and policy
        if self.args.vae_mixture_num > 1:
            if self.args.vae_extrapolate:
                self.vae = VaribadVAEMixtureExt(self.args, self.logger, lambda: self.iter_idx)
            else:
                self.vae = VaribadVAEMixture(self.args, self.logger, lambda: self.iter_idx)
        else:
            self.vae = VaribadVAE(self.args, self.logger, lambda: self.iter_idx)
        self.policy_storage = self.initialise_policy_storage()
        self.policy = self.initialise_policy()
        self.policy_resample = PolicyResample(self.args, self.args.state_dim, self.args.latent_dim,
                                              self.args.vae_mixture_num).to(device)


    def initialise_policy_storage(self):
        return OnlineStorage(args=self.args,
                             num_steps=self.args.policy_num_steps,
                             num_processes=self.args.num_processes,
                             state_dim=self.args.state_dim,
                             latent_dim=self.args.latent_dim,
                             belief_dim=self.args.belief_dim,
                             task_dim=self.args.task_dim,
                             prob_dim=self.args.vae_mixture_num,
                             action_space=self.args.action_space,
                             hidden_size=self.args.encoder_gru_hidden_size,
                             normalise_rewards=self.args.norm_rew_for_policy,
                             )

    def initialise_policy(self):

        # initialise policy network
        policy_net = Policy(
            args=self.args,
            #
            pass_state_to_policy=self.args.pass_state_to_policy,
            pass_latent_to_policy=self.args.pass_latent_to_policy,
            pass_belief_to_policy=self.args.pass_belief_to_policy,
            pass_task_to_policy=self.args.pass_task_to_policy,
            pass_prob_to_policy=self.args.pass_prob_to_policy,
            dim_state=self.args.state_dim,
            dim_latent=self.args.latent_dim * 2,
            dim_belief=self.args.belief_dim,
            dim_task=self.args.task_dim,
            #
            hidden_layers=self.args.policy_layers,
            activation_function=self.args.policy_activation_function,
            policy_initialisation=self.args.policy_initialisation,
            #
            action_space=self.envs.action_space,
            init_std=self.args.policy_init_std,
            min_std=self.args.policy_min_std,
            max_std=self.args.policy_max_std
        ).to(device)

        # initialise policy trainer
        if self.args.policy == 'a2c':
            policy = A2C(
                self.args,
                policy_net,
                self.args.policy_value_loss_coef,
                self.args.policy_entropy_coef,
                policy_optimiser=self.args.policy_optimiser,
                policy_anneal_lr=self.args.policy_anneal_lr,
                train_steps=self.num_updates,
                optimiser_vae=self.vae.optimiser_vae,
                lr=self.args.lr_policy,
                eps=self.args.policy_eps,
            )
        elif self.args.policy == 'ppo':
            if self.args.ppo_disc:
                policy = PPO_DISC(
                    self.args,
                    policy_net,
                    self.args.policy_value_loss_coef,
                    self.args.policy_entropy_coef,
                    policy_optimiser=self.args.policy_optimiser,
                    policy_anneal_lr=self.args.policy_anneal_lr,
                    train_steps=self.num_updates,
                    lr=self.args.lr_policy,
                    eps=self.args.policy_eps,
                    ppo_epoch=self.args.ppo_num_epochs,
                    num_mini_batch=self.args.ppo_num_minibatch,
                    use_huber_loss=self.args.ppo_use_huberloss,
                    use_clipped_value_loss=self.args.ppo_use_clipped_value_loss,
                    clip_param=self.args.ppo_clip_param,
                    optimiser_vae=self.vae.optimiser_vae,
                )
            else:
                policy = PPO(
                    self.args,
                    policy_net,
                    self.args.policy_value_loss_coef,
                    self.args.policy_entropy_coef,
                    policy_optimiser=self.args.policy_optimiser,
                    policy_anneal_lr=self.args.policy_anneal_lr,
                    train_steps=self.num_updates,
                    lr=self.args.lr_policy,
                    eps=self.args.policy_eps,
                    ppo_epoch=self.args.ppo_num_epochs,
                    num_mini_batch=self.args.ppo_num_minibatch,
                    use_huber_loss=self.args.ppo_use_huberloss,
                    use_clipped_value_loss=self.args.ppo_use_clipped_value_loss,
                    clip_param=self.args.ppo_clip_param,
                    optimiser_vae=self.vae.optimiser_vae,
                )
        else:
            raise NotImplementedError

        return policy

    def train(self):
        """ Main Meta-Training loop """
        start_time = time.time()

        # reset environments
        prev_state, belief, task = utl.reset_env(self.envs, self.args)
        if self.args.virtual_intrinsic > 0.0:
            y_intercept = self.sample_y(num_virtual_skills=self.args.num_virtual_skills, include_smaller=self.args.include_smaller, dist=self.args.virtual_dist)

        # insert initial observation / embeddings to rollout storage
        self.policy_storage.prev_state[0].copy_(prev_state)

        # log once before training
        if self.args.load_iter is None:
            iter_scope = np.arange(-1,5000, 200)
        else:
            iter_scope = [int(self.args.load_iter)]
        for iter_idx in iter_scope:
            self.frames = (iter_idx + 1) * self.args.policy_num_steps * self.args.num_processes
            self.iter_idx = iter_idx
            if self.args.load_dir is not None:
                print('loading pretrained model from ', self.args.load_dir)
                self.policy.actor_critic = torch.load(self.args.load_dir + '/models/policy{}.pt'.format(iter_idx))
                self.policy.actor_critic.train()
                self.vae.encoder = torch.load(self.args.load_dir + '/models/encoder{}.pt'.format(iter_idx))
                self.vae.encoder.train()
                if self.vae.state_decoder is not None:
                    self.vae.state_decoder = torch.load(self.args.load_dir + '/models/state_decoder{}.pt'.format(iter_idx))
                    self.vae.state_decoder.train()
                if self.vae.reward_decoder is not None:
                    self.vae.reward_decoder = torch.load(self.args.load_dir + '/models/reward_decoder{}.pt'.format(iter_idx))
                    self.vae.reward_decoder.train()
                if self.vae.task_decoder is not None:
                    self.vae.task_decoder = torch.load(self.args.load_dir + '/models/task_decoder{}.pt'.format(iter_idx))
                    self.vae.task_decoder.train()
                self.vae.optimiser_vae.load_state_dict(torch.load(self.args.load_dir + '/models/optimiser_vae{}.pt'.format(iter_idx)))
                self.policy.optimiser.load_state_dict(torch.load(self.args.load_dir + '/models/optimiser_pol{}.pt'.format(iter_idx)))

                if self.args.norm_rew_for_policy:
                    rew_rms = utl.load_obj(self.args.load_dir + 'models/', 'env_rew_rms{}'.format(iter_idx))
                    self.envs.venv.ret_rms = rew_rms
                if self.args.norm_state_for_policy:
                    obs_rms = utl.load_obj(self.args.load_dir + 'models/', 'pol_state_rms{}'.format(iter_idx))
                    self.policy.actor_critic.state_rms = obs_rms

            with torch.no_grad():
                self.log(None, None, start_time)

        self.envs.close()

    def log(self, run_stats, train_stats, start_time):

        # --- visualise behaviour of policy ---

        # --- evaluate policy ----

        #if self.iter_idx>0 and ((self.iter_idx + 1) % self.args.eval_interval == 0):
        if 1:
            os.makedirs('{}/{}'.format(self.logger.full_output_folder, self.iter_idx))
            ret_rms = None #we don't need normalised reward for eval
            total_parametric_num = self.args.parametric_num

            num_worker = 10
            returns_array = np.zeros((15, total_parametric_num, self.args.max_rollouts_per_task))
            latent_means_array = np.zeros((15, total_parametric_num, self.args.latent_dim))
            latent_logvars_array = np.zeros((15, total_parametric_num, self.args.latent_dim))

            successes_array = np.zeros((15, total_parametric_num))
            save_episode_successes = True
            if save_episode_successes:
                episode_successes_array = np.zeros((15, total_parametric_num, self.args.max_rollouts_per_task))

            save_episode_probs = False

            #save_episode_probs = (self.iter_idx + 1) % (20 * self.args.eval_interval) == 0
            probs_array = np.zeros((15, total_parametric_num, self.args.vae_mixture_num))
            if save_episode_probs:
                episode_probs_array = np.zeros((15, total_parametric_num, self.args.max_rollouts_per_task,
                                                self.envs._max_episode_steps, self.args.vae_mixture_num))

            for task_class in range(15):
                print(self.iter_idx, task_class)
                for parametric_num in range(total_parametric_num // num_worker):
                    task_list = np.concatenate((np.expand_dims(np.repeat(task_class, num_worker), axis=1),
                                                np.expand_dims(np.arange(num_worker * parametric_num,
                                                                         num_worker * (parametric_num + 1)), axis=1)),axis=1)

                    returns_per_episode, latent_mean, latent_logvar, successes,  prob, episode_probs, episode_successes = utl_eval.evaluate_metaworld(
                        args=self.args,
                        policy=self.policy,
                        ret_rms=ret_rms,
                        encoder=self.vae.encoder,
                        iter_idx=self.iter_idx,
                        tasks=None,
                        test=False,
                        task_list=task_list,
                        save_episode_probs=save_episode_probs,
                        save_episode_successes=save_episode_successes,
                        )

                    returns_array[task_class, parametric_num * num_worker:(parametric_num + 1) * num_worker, :] = returns_per_episode
                    latent_means_array[task_class, parametric_num * num_worker:(parametric_num + 1) * num_worker, :] = latent_mean
                    latent_logvars_array[task_class, parametric_num * num_worker:(parametric_num + 1) * num_worker, :] = latent_logvar
                    successes_array[task_class, parametric_num * num_worker:(parametric_num + 1) * num_worker] = successes

                    probs_array[task_class, parametric_num * num_worker:(parametric_num + 1) * num_worker, :] = prob
                    if save_episode_probs:
                        episode_probs_array[task_class, parametric_num * num_worker:(parametric_num + 1) * num_worker, :, :, :] = episode_probs
                    if save_episode_successes:
                        episode_successes_array[task_class, parametric_num * num_worker:(parametric_num + 1) * num_worker, :] = episode_successes

            taskwise_mean_return = np.mean(np.mean(returns_array, axis=2), axis=1)
            taskwise_mean_final_return = np.mean(returns_array[:,:,-1], axis=1)
            taskwise_mean_success = np.mean(successes_array, axis=1)
            taskwise_mean_final_success = np.mean(episode_successes_array[:,:,-1], axis=1)

            print(f"Updates {self.iter_idx}, "
                  f"Frames {self.frames}, "
                  f"FPS {int(self.frames / (time.time() - start_time))}, \n"
                  f" Mean return per episode (train): {np.mean(taskwise_mean_return[:10])},"
                  f" Mean return per episode (test): {np.mean(taskwise_mean_return[10:])},\n"
                  f" Mean final return per episode (train): {np.mean(taskwise_mean_final_return[:10])},"
                  f" Mean final return per episode (test): {np.mean(taskwise_mean_final_return[10:])},\n"
                  f" Mean success rate (train): {np.mean(taskwise_mean_success[:10])},"
                  f" Mean final success rate (train): {np.mean(taskwise_mean_final_success[:10])},\n"
                  f" Mean success rate (test): {np.mean(taskwise_mean_success[10:])}"
                  f" Mean final success rate (test): {np.mean(taskwise_mean_final_success[10:])}"
                  )
            print("train taskwise success rates: ", taskwise_mean_success[:10])
            print("train taskwise final success rates: ", taskwise_mean_final_success[:10])
            print("test taskwise success rates: ", taskwise_mean_success[10:])
            print("test taskwise final success rates: ", taskwise_mean_final_success[10:])

            with open(self.logger.full_output_folder + '/log_eval.csv', 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(np.concatenate(([self.iter_idx, int(self.frames)], taskwise_mean_return, taskwise_mean_success, taskwise_mean_final_success, taskwise_mean_final_return)))

            np.save('{}/{}/returns.npy'.format(self.logger.full_output_folder, self.iter_idx), returns_array)
            np.save('{}/{}/latent_means.npy'.format(self.logger.full_output_folder, self.iter_idx), latent_means_array)
            np.save('{}/{}/latent_logvars.npy'.format(self.logger.full_output_folder, self.iter_idx),
                    latent_logvars_array)
            np.save('{}/{}/successes.npy'.format(self.logger.full_output_folder, self.iter_idx), successes_array)
            if save_episode_successes:
                np.save('{}/{}/episode_successes_array.npy'.format(self.logger.full_output_folder, self.iter_idx),
                        episode_successes_array)

            np.save('{}/{}/probs.npy'.format(self.logger.full_output_folder, self.iter_idx), probs_array)
            if save_episode_probs:
                np.save('{}/{}/episode_probs_array.npy'.format(self.logger.full_output_folder, self.iter_idx),
                        episode_probs_array)