import os
import time

import gym
import numpy as np
import torch

from algorithms.online_storage import OnlineStorage
from algorithms.ppo import PPO
from environments.parallel_envs import make_vec_envs
from models.policy import Policy
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


class MetaLearnerML10LDM:
    """
    Meta-Learner class with the main training loop for variBAD.
    """
    def __init__(self, args):

        self.args = args
        self.args.pass_prob_to_policy = False
        self.virtual_ratio = self.args.virtual_ratio
        utl.seed(self.args.seed, self.args.deterministic_execution)

        # calculate number of updates and keep count of frames/iterations
        self.num_updates = int(args.num_frames) // args.policy_num_steps // args.num_processes
        if self.args.load_dir is None:
            self.frames = 0
            self.iter_idx = -1
        else:
            self.frames = (int(self.args.load_iter)+1) * self.args.policy_num_steps * self.args.num_processes
            self.iter_idx = int(self.args.load_iter)
        self.recent_train_success = np.zeros(10)
        self.task_count = np.zeros(10)
        #self.return_list = torch.zeros((self.args.num_processes)).to(device)

        # initialise tensorboard logger
        self.logger = TBLogger(self.args, self.args.exp_label)

        header = ['iter', 'frames', 'R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14',
                  'S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14',
                  'SF0', 'SF1', 'SF2', 'SF3', 'SF4', 'SF5', 'SF6', 'SF7', 'SF8', 'SF9', 'SF10', 'SF11', 'SF12', 'SF13', 'SF14',
                  'RF0', 'RF1', 'RF2', 'RF3', 'RF4', 'RF5', 'RF6', 'RF7', 'RF8', 'RF9', 'RF10', 'RF11', 'RF12', 'RF13', 'RF14',
                  ]
        with open(self.logger.full_output_folder+'/log_eval.csv', 'w', encoding='UTF8') as f:
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
        self.vae = VaribadVAE(self.args, self.logger, lambda: self.iter_idx)
        self.policy_storage = self.initialise_policy_storage()
        self.policy = self.initialise_policy()

        if self.args.load_dir is not None:
            print('loading pretrained model from ', self.args.load_dir)
            self.policy.actor_critic = torch.load(self.args.load_dir+'/models/policy{}.pt'.format(self.args.load_iter))
            self.policy.actor_critic.train()
            self.vae.encoder = torch.load(self.args.load_dir+'/models/encoder{}.pt'.format(self.args.load_iter))
            self.vae.encoder.train()
            if self.vae.state_decoder is not None:
                self.vae.state_decoder = torch.load(self.args.load_dir+'/models/state_decoder{}.pt'.format(self.args.load_iter))
                self.vae.state_decoder.train()
            if self.vae.reward_decoder is not None:
                self.vae.reward_decoder = torch.load(self.args.load_dir+'/models/reward_decoder{}.pt'.format(self.args.load_iter))
                self.vae.reward_decoder.train()
            if self.vae.task_decoder is not None:
                self.vae.task_decoder = torch.load(self.args.load_dir+'/models/task_decoder{}.pt'.format(self.args.load_iter))
                self.vae.task_decoder.train()
            self.vae.optimiser_vae.load_state_dict(torch.load(self.args.load_dir+'/models/optimiser_vae{}.pt'.format(self.args.load_iter)))
            self.policy.optimiser.load_state_dict(torch.load(self.args.load_dir+'/models/optimiser_pol{}.pt'.format(self.args.load_iter)))

            if self.args.norm_rew_for_policy:
                rew_rms = utl.load_obj(self.args.load_dir + 'models/', 'env_rew_rms{}'.format(self.args.load_iter))
                self.envs.venv.ret_rms = rew_rms
            if self.args.norm_state_for_policy:
                obs_rms = utl.load_obj(self.args.load_dir + 'models/', 'pol_state_rms{}'.format(self.args.load_iter))
                self.policy.actor_critic.state_rms = obs_rms

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
        if self.args.policy == 'ppo':
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

    def sample_z(self, num_procs, latent_dim, past_z):
        with torch.no_grad():
            z_sample = torch.zeros(num_procs, latent_dim)
            if past_z == None:
                z_intercept = torch.zeros(num_procs, latent_dim).to(device)
            else:
                for i in range(num_procs):
                    if self.args.include_smaller:
                        index = random.sample(range(num_procs), random.choice(range(1, self.args.num_virtual_skills + 1)))
                    else:
                        index = random.sample(range(num_procs), self.args.num_virtual_skills)
                    alpha = 1e-20 * torch.ones(num_procs)
                    alpha[index] = 1.0
                    z_dist = torch.distributions.dirichlet.Dirichlet(alpha)
                    z_sample[i, :] = torch.matmul(z_dist.sample(), past_z)
                z_intercept = past_z.clone()
        return z_intercept.to(device)

    def train(self):
        """ Main Meta-Training loop """
        start_time = time.time()

        # reset environments
        prev_state, belief, task = utl.reset_env(self.envs, self.args)
        z_intercept = self.sample_z(num_procs=self.args.num_processes, latent_dim = self.args.latent_dim, past_z = None)
        # insert initial observation / embeddings to rollout storage
        self.policy_storage.prev_state[0].copy_(prev_state)

        # log once before training
        with torch.no_grad():
            self.log(None, None, start_time)
        self.iter_idx += 1 # number of interactions with the real environment
        self.virtual_iter_idx = self.iter_idx #total interactions including the virtual
        while self.iter_idx < self.num_updates:
            if random.random()<self.virtual_ratio: #this code is valid only when policy_num_steps == 5000
                virtual = True
            else:
                virtual = False
            # First, re-compute the hidden states given the current rollouts (since the VAE might've changed)
            with torch.no_grad():
                latent_sample, latent_mean, latent_logvar, hidden_state = self.encode_running_trajectory(virtual)
                y=prob=None

            # add this initial hidden state to the policy storage
            assert len(self.policy_storage.latent_mean) == 0  # make sure we emptied buffers
            self.policy_storage.hidden_states[0].copy_(hidden_state)
            self.policy_storage.latent_samples.append(latent_sample.clone())
            self.policy_storage.latent_mean.append(latent_mean.clone())
            self.policy_storage.latent_logvar.append(latent_logvar.clone())

            # rollout policies for a few steps
            for step in range(self.args.policy_num_steps):
                #print(self.iter_idx, step)
                # sample actions from policy
                with torch.no_grad():
                    value, action = utl.select_action(
                        args=self.args,
                        policy=self.policy,
                        state=prev_state,
                        belief=belief,
                        task=task,
                        prob=prob,
                        deterministic=False,
                        latent_sample=latent_sample,
                        latent_mean=latent_mean,
                        latent_logvar=latent_logvar,
                    )

                [next_state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(self.envs, action, self.args)

                # take step in the environment
                if virtual: #use virtual environment
                    with torch.no_grad():
                        rew_raw_pred = self.vae.reward_decoder(z_intercept.detach(), next_state, prev_state, action)
                    rew_raw = torch.clamp(rew_raw_pred.clone().detach(), min=0.0, max=10.0)
                    rew_raw_np = rew_raw.cpu().numpy()
                    rew_normalised = self.envs.venv._rewfilt2(rew_raw_np)
                    rew_normalised = torch.from_numpy(rew_normalised)[:,0].unsqueeze(dim=1).float().to(device)
                done = torch.from_numpy(np.array(done, dtype=int)).to(device).float().view((-1, 1))
                # create mask for episode ends
                masks_done = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)
                # bad_mask is true if episode ended because time limit was reached
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos]).to(device)

                with torch.no_grad():
                    # compute next embedding (for next loop and/or value prediction bootstrap)
                    latent_sample, latent_mean, latent_logvar, hidden_state  = utl.update_encoding(
                        encoder=self.vae.encoder,
                        next_obs=next_state,
                        action=action,
                        reward=rew_raw,
                        done=done,
                        hidden_state=hidden_state,
                        vae_mixture_num=self.args.vae_mixture_num)

                # before resetting, update the embedding and add to vae buffer
                # (last state might include useful task info)
                if not (self.args.disable_decoder and self.args.disable_kl_term):
                    #if not virtual: #do not insert virtual in the vae buffer? maybe we have to?
                    if not virtual:
                        self.vae.rollout_storage.insert(prev_state.clone(),
                                                        action.detach().clone(),
                                                        next_state.clone(),
                                                        rew_raw.clone(),
                                                        done.clone(),
                                                        task.clone() if task is not None else None)

                # add the obs before reset to the policy storage
                self.policy_storage.next_state[step] = next_state.clone()

                # reset environments that are done
                done_indices = np.argwhere(done.cpu().flatten()).flatten()

                if len(done_indices) > 0:
                    task_indicies = np.array(self.envs.get_task())[:,0]
                    for i in range(10):
                        self.task_count[i] += np.count_nonzero(task_indicies == i)
                    #TODO1: for virtual envs, we need to store the initial states in a buffer
                    next_state, belief, task = utl.reset_env(self.envs, self.args, indices=done_indices, state=next_state)

                    if not virtual: #resample z with new distribution
                        z_intercept = self.sample_z(num_procs=self.args.num_processes, latent_dim = self.args.latent_dim, past_z = latent_sample.cpu())


                # TODO: deal with resampling for posterior sampling algorithm
                #     latent_sample = latent_sample
                #     latent_sample[i] = latent_sample[i]

                # add experience to policy buffer
                self.policy_storage.insert(
                    state=next_state,
                    belief=belief,
                    task=task,
                    actions=action,
                    rewards_raw=rew_raw,
                    rewards_normalised=rew_normalised,
                    value_preds=value,
                    masks=masks_done,
                    bad_masks=bad_masks,
                    done=done,
                    hidden_states=hidden_state.squeeze(0),
                    latent_sample=latent_sample,
                    latent_mean=latent_mean,
                    latent_logvar=latent_logvar,
                    y=y,
                    prob=prob,
                )
                prev_state = next_state

                self.frames += self.args.num_processes

            # --- UPDATE ---

            if self.args.precollect_len <= self.frames:

                # check if we are pre-training the VAE
                if self.args.pretrain_len > self.iter_idx:
                    for p in range(self.args.num_vae_updates_per_pretrain):
                        self.vae.compute_vae_loss(update=True,
                                                  pretrain_index=self.iter_idx * self.args.num_vae_updates_per_pretrain + p)
                # otherwise do the normal update (policy + vae)
                else:

                    train_stats = self.update(state=prev_state,
                                              belief=belief,
                                              task=task,
                                              prob=prob,
                                              latent_sample=latent_sample,
                                              latent_mean=latent_mean,
                                              latent_logvar=latent_logvar)

                    # log
                    run_stats = [action, self.policy_storage.action_log_probs, value]

                    with torch.no_grad():
                        self.log(run_stats, train_stats, start_time)

            # clean up after update
            self.policy_storage.after_update()

            self.virtual_iter_idx+=1
            self.iter_idx +=1
            self.virtual_ratio += self.args.virtual_ratio_increment*(self.args.num_processes*self.args.policy_num_steps)/1e8

        self.envs.close()

    def encode_running_trajectory(self, virtual = False):
        """
        (Re-)Encodes (for each process) the entire current trajectory.
        Returns sample/mean/logvar and hidden state (if applicable) for the current timestep.
        :return:
        """

        # for each process, get the current batch (zero-padded obs/act/rew + length indicators)
        prev_obs, next_obs, act, rew, lens = self.vae.rollout_storage.get_running_batch()

        # get embedding - will return (1+sequence_len) * batch * input_size -- includes the prior!
        all_latent_samples, all_latent_means, all_latent_logvars, all_hidden_states = self.vae.encoder(actions=act,
                                                                                                       states=next_obs,
                                                                                                       rewards=rew,
                                                                                                       hidden_state=None,
                                                                                                       return_prior=True)

        # get the embedding / hidden state of the current time step (need to do this since we zero-padded)
        latent_sample = (torch.stack([all_latent_samples[lens[i]][i] for i in range(len(lens))])).to(device)
        latent_mean = (torch.stack([all_latent_means[lens[i]][i] for i in range(len(lens))])).to(device)
        latent_logvar = (torch.stack([all_latent_logvars[lens[i]][i] for i in range(len(lens))])).to(device)
        hidden_state = (torch.stack([all_hidden_states[lens[i]][i] for i in range(len(lens))])).to(device)

        return latent_sample, latent_mean, latent_logvar, hidden_state

    def get_value(self, state, belief, task, prob, latent_sample, latent_mean, latent_logvar):
        latent = utl.get_latent_for_policy(self.args, latent_sample=latent_sample, latent_mean=latent_mean, latent_logvar=latent_logvar)
        return self.policy.actor_critic.get_value(state=state, belief=belief, task=task, latent=latent, prob=prob).detach()

    def update(self, state, belief, task,prob, latent_sample, latent_mean, latent_logvar):
        """
        Meta-update.
        Here the policy is updated for good average performance across tasks.
        :return:
        """
        # update policy (if we are not pre-training, have enough data in the vae buffer, and are not at iteration 0)
        if self.iter_idx >= self.args.pretrain_len and self.iter_idx > 0:

            # bootstrap next value prediction
            with torch.no_grad():
                next_value = self.get_value(state=state,
                                            belief=belief,
                                            task=task,
                                            prob=prob,
                                            latent_sample=latent_sample,
                                            latent_mean=latent_mean,
                                            latent_logvar=latent_logvar)

            # compute returns for current rollouts
            self.policy_storage.compute_returns(next_value, self.args.policy_use_gae, self.args.policy_gamma,
                                                self.args.policy_tau,
                                                use_proper_time_limits=self.args.use_proper_time_limits)

            # update agent (this will also call the VAE update!)
            policy_train_stats = self.policy.update(
                policy_storage=self.policy_storage,
                encoder=self.vae.encoder,
                rlloss_through_encoder=self.args.rlloss_through_encoder,
                compute_vae_loss=self.vae.compute_vae_loss)
        else:
            policy_train_stats = 0, 0, 0, 0

            # pre-train the VAE
            if self.iter_idx < self.args.pretrain_len:
                self.vae.compute_vae_loss(update=True)

        return policy_train_stats

    def log(self, run_stats, train_stats, start_time):

        # --- visualise behaviour of policy ---

        # --- evaluate policy ----

        if (self.iter_idx + 1) % self.args.eval_interval == 0:
            os.makedirs('{}/{}'.format(self.logger.full_output_folder, self.iter_idx))
            ret_rms = None #we don't need normalised reward for eval
            if (self.iter_idx + 1) % (10 * self.args.eval_interval) == 0:
                total_parametric_num = 10
            else:
                total_parametric_num = 10

            num_worker = 10
            returns_array = np.zeros((15, total_parametric_num, self.args.max_rollouts_per_task))
            latent_means_array = np.zeros((15, total_parametric_num, self.args.latent_dim))
            latent_logvars_array = np.zeros((15, total_parametric_num, self.args.latent_dim))

            successes_array = np.zeros((15, total_parametric_num))
            save_episode_successes = True
            if save_episode_successes:
                episode_successes_array = np.zeros((15, total_parametric_num, self.args.max_rollouts_per_task))

            save_episode_probs = False

            for task_class in range(15):
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
            print("history: ", self.task_count)
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
            np.save('{}/{}/task_count.npy'.format(self.logger.full_output_folder, self.iter_idx), self.task_count)
            if save_episode_successes:
                np.save('{}/{}/episode_successes_array.npy'.format(self.logger.full_output_folder, self.iter_idx),
                        episode_successes_array)
            self.task_count = np.zeros((self.args.num_processes))
            self.recent_train_success = taskwise_mean_success[:10]
        # --- save models ---
        if (self.iter_idx + 1) % self.args.save_interval == 0:
            save_path = os.path.join(self.logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            idx_labels = ['']
            if self.args.save_intermediate_models:
                idx_labels.append(int(self.iter_idx))

            for idx_label in idx_labels:

                torch.save(self.policy.actor_critic, os.path.join(save_path, f"policy{idx_label}.pt"))
                torch.save(self.vae.encoder, os.path.join(save_path, f"encoder{idx_label}.pt"))
                #torch.save(self.policy_resample, os.path.join(save_path, f"policy_resample{idx_label}.pt"))
                if self.vae.state_decoder is not None:
                    torch.save(self.vae.state_decoder, os.path.join(save_path, f"state_decoder{idx_label}.pt"))
                if self.vae.reward_decoder is not None:
                    torch.save(self.vae.reward_decoder, os.path.join(save_path, f"reward_decoder{idx_label}.pt"))
                if self.vae.task_decoder is not None:
                    torch.save(self.vae.task_decoder, os.path.join(save_path, f"task_decoder{idx_label}.pt"))
                torch.save(self.vae.optimiser_vae.state_dict(), os.path.join(save_path, f"optimiser_vae{idx_label}.pt"))
                torch.save(self.policy.optimiser.state_dict(), os.path.join(save_path, f"optimiser_pol{idx_label}.pt"))
                #torch.save(self.policy_resample.optimiser.state_dict(),os.path.join(save_path, f"optimiser_pol_res{idx_label}.pt"))
                # save normalisation params of envs
                if self.args.norm_rew_for_policy:
                    rew_rms = self.envs.venv.ret_rms
                    utl.save_obj(rew_rms, save_path, f"env_rew_rms{idx_label}")
                # TODO: grab from policy and save?
                if self.args.norm_state_for_policy:
                    obs_rms = self.policy.actor_critic.state_rms
                    utl.save_obj(obs_rms, save_path, f"pol_state_rms{idx_label}")


        # --- log some other things ---

        if ((self.iter_idx + 1) % self.args.log_interval == 0) and (train_stats is not None):

            self.logger.add('environment/state_max', self.policy_storage.prev_state.max(), self.iter_idx)
            self.logger.add('environment/state_min', self.policy_storage.prev_state.min(), self.iter_idx)

            self.logger.add('environment/rew_max', self.policy_storage.rewards_raw.max(), self.iter_idx)
            self.logger.add('environment/rew_min', self.policy_storage.rewards_raw.min(), self.iter_idx)

            self.logger.add('policy_losses/value_loss', train_stats[0], self.iter_idx)
            self.logger.add('policy_losses/action_loss', train_stats[1], self.iter_idx)
            self.logger.add('policy_losses/dist_entropy', train_stats[2], self.iter_idx)
            self.logger.add('policy_losses/sum', train_stats[3], self.iter_idx)

            self.logger.add('policy/action', run_stats[0][0].float().mean(), self.iter_idx)
            if hasattr(self.policy.actor_critic, 'logstd'):
                self.logger.add('policy/action_logstd', self.policy.actor_critic.dist.logstd.mean(), self.iter_idx)
            self.logger.add('policy/action_logprob', run_stats[1].mean(), self.iter_idx)
            self.logger.add('policy/value', run_stats[2].mean(), self.iter_idx)

            self.logger.add('encoder/latent_mean', torch.cat(self.policy_storage.latent_mean).mean(), self.iter_idx)
            self.logger.add('encoder/latent_logvar', torch.cat(self.policy_storage.latent_logvar).mean(), self.iter_idx)

            # log the average weights and gradients of all models (where applicable)
            for [model, name] in [
                [self.policy.actor_critic, 'policy'],
                [self.vae.encoder, 'encoder'],
                [self.vae.reward_decoder, 'reward_decoder'],
                [self.vae.state_decoder, 'state_transition_decoder'],
                [self.vae.task_decoder, 'task_decoder']
            ]:
                if model is not None:
                    param_list = list(model.parameters())
                    param_mean = np.mean([param_list[i].data.cpu().numpy().mean() for i in range(len(param_list))])

                    #print('name', name)
                    #print('model', model)
                    #for i in range(len(param_list)):
                    #    print('param_list grad ',i, param_list[i].grad is None , param_list[i].size())

                    self.logger.add('weights/{}'.format(name), param_mean, self.iter_idx)
                    if name == 'policy':
                        self.logger.add('weights/policy_std', param_list[0].data.mean(), self.iter_idx)
                    if param_list[0].grad is not None:
                        param_grad_mean = np.mean([param_list[i].grad.cpu().numpy().mean() for i in range(len(param_list))])
                        self.logger.add('gradients/{}'.format(name), param_grad_mean, self.iter_idx)

