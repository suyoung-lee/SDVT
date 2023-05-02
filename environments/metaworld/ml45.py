import matplotlib as mpl
import random
import numpy as np
import torch
from utils import helpers as utl
import matplotlib.pyplot as plt
import seaborn as sns

from gym import Env
from gym import spaces

import metaworld


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ML45(Env):

    def __init__(self, max_episode_steps=500, SEED=10):
        ml45 = metaworld.ML45(seed=SEED)
        self.SEED = SEED
        self.train_env_name_list = [name for name,_ in ml45.train_classes.items()]
        self.train_env_cls_list = [env_cls() for _,env_cls in ml45.train_classes.items()]
        self.test_env_name_list = [name for name,_ in ml45.test_classes.items()]
        self.test_env_cls_list = [env_cls() for _,env_cls in ml45.test_classes.items()]
        self.train_tasks = ml45.train_tasks
        self.test_tasks = ml45.test_tasks

        self.reset_task()
        self.task_dim = 2

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        self._max_episode_steps = max_episode_steps

    def set_task(self, task):
        self._task = task

    def get_task(self):
        return self._task

    def reset_task(self, task=None):
        if task is None:
            env_ind = random.choice(range(45))
            #env_ind = random.choice([x for x in range(0, 45) if x != 34]) #observation explosion bug in pegunplugside
            self._env = self.train_env_cls_list[env_ind]
            _env_name = self.train_env_name_list[env_ind]
            subtask_ind = random.choice(range(50))
            self.set_task([env_ind, subtask_ind])
            self._env.set_task([_task for _task in self.train_tasks
                                  if _task.env_name == _env_name][subtask_ind])
        else: #two dimensional task
            env_ind = task[0]
            subtask_ind = task[1]
            self.set_task([env_ind, subtask_ind])
            if env_ind <45:
                self._env = self.train_env_cls_list[env_ind]
                _env_name = self.train_env_name_list[env_ind]
                self._env.set_task([_task for _task in self.train_tasks
                                      if _task.env_name == _env_name][subtask_ind])
            else:
                self._env = self.test_env_cls_list[env_ind-45]
                _env_name = self.test_env_name_list[env_ind-45]
                self._env.set_task([_task for _task in self.test_tasks
                                      if _task.env_name == _env_name][subtask_ind])
        #print('seed:', self.SEED, ', index: ', env_ind, subtask_ind, ', target', self._env._target_pos, ', object', self._env.obj_init_pos)
        self.reset()
        return self._state

    def _reset_model(self):
        # resetting to unwrapped metaworld initial position not the task type
        self._state = self._env.reset()
        #print('reset_model call end ', self.get_task())
        return self._get_obs()

    def reset(self, task=None):
        if task is not None:
            self.reset_task(task)
        #print('reset call end ',task, self.get_task())
        return self._reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    # add 'image' to render
    def step(self, action):


        action_ = np.clip(action, self.action_space.low, self.action_space.high)
        self._state, reward, done, info = self._env.step(action_)
        ob = self._get_obs()
        ob = np.clip(ob, -1.0, 1.0) #mitigate instability due to bugs with observation exploration in some tasks in ML45
        #info = {'task': self.get_task(), 'success': info['success'], 'image': self._env.render(offscreen=True)}  # for rendering
        info = {'task': self.get_task(), 'success': info['success']}

        return ob, reward, done, info


    @staticmethod
    def visualise_behaviour_ml10(env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            image_folder=None,
                            task_num = None,
                            args_pol = None,
                            **kwargs
                            ):

        num_episodes = args.max_rollouts_per_task

        # --- initialise things we want to keep track of ---

        episode_prev_obs = [[] for _ in range(num_episodes)]
        episode_next_obs = [[] for _ in range(num_episodes)]
        episode_actions = [[] for _ in range(num_episodes)]
        episode_rewards = [[] for _ in range(num_episodes)]
        episode_successes = [[] for _ in range(num_episodes)]

        episode_returns = []
        episode_lengths = []

        if encoder is not None:
            episode_latent_samples = [[] for _ in range(num_episodes)]
            episode_latent_means = [[] for _ in range(num_episodes)]
            episode_latent_logvars = [[] for _ in range(num_episodes)]
        else:
            episode_latent_samples = episode_latent_means = episode_latent_logvars = None

        # --- roll out policy ---

        # (re)set environment

        env.reset_task([task_num[0], task_num[1]])
        state = env.reset(task = [[task_num[0], task_num[1]]])
        belief = None
        task = None

        start_obs_raw = state.clone()
        task = task.view(-1) if task is not None else None

        # initialise actions and rewards (used as initial input to policy if we have a recurrent policy)
        if hasattr(args, 'hidden_size'):
            hidden_state = torch.zeros((1, args.hidden_size)).to(device)
        else:
            hidden_state = None

        # keep track of what task we're in and the position of the cheetah
        pos = [[] for _ in range(args.max_rollouts_per_task)]
        start_pos = state

        for episode_idx in range(num_episodes):

            curr_rollout_rew = []
            pos[episode_idx].append(start_pos[0])

            if episode_idx == 0:
                if encoder is not None:
                    # reset to prior
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
                    curr_latent_sample = curr_latent_sample[0].to(device)
                    curr_latent_mean = curr_latent_mean[0].to(device)
                    curr_latent_logvar = curr_latent_logvar[0].to(device)
                else:
                    curr_latent_sample = curr_latent_mean = curr_latent_logvar = None

            if encoder is not None:
                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_obs_raw.clone())
                else:
                    episode_prev_obs[episode_idx].append(state.clone())
                # act
                latent = utl.get_latent_for_policy(args,
                                                   latent_sample=curr_latent_sample,
                                                   latent_mean=curr_latent_mean,
                                                   latent_logvar=curr_latent_logvar)
                _, action = policy.act(state=state.view(-1), latent=latent, belief=belief, task=task,
                                       deterministic=True)

                (state, belief, task), (rew, rew_normalised), done, info = utl.env_step(env, action, args)
                state = state.float().reshape((1, -1)).to(device)
                task = task.view(-1) if task is not None else None

                # keep track of position
                pos[episode_idx].append(state[0])

                if encoder is not None:
                    # update task embedding
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                        action.reshape(1, -1).float().to(device), state, rew.reshape(1, -1).float().to(device),
                        hidden_state, return_prior=False)

                    episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                    episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                    episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.clone())
                episode_successes[episode_idx].append(info[0]['success'])
                curr_rollout_rew.append(rew.clone())

                if info[0]['done_mdp'] and not done:
                    start_obs_raw = info[0]['start_state']
                    start_obs_raw = torch.from_numpy(start_obs_raw).float().reshape((1, -1)).to(device)
                    start_pos = start_obs_raw
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)

        # clean up
        if encoder is not None:
            episode_latent_means = [torch.stack(e) for e in episode_latent_means]
            episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.stack(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]


        if image_folder is not None:
            behaviour_dir = '{}/{}/{:02d}_{:02d}'.format(image_folder, iter_idx, task_num[0],task_num[1])

            for i in range(num_episodes):
                np.savez(behaviour_dir + '_' + str(i) + '_data.npz',
                         episode_latent_means=episode_latent_means[i].detach().cpu().numpy(),
                         episode_latent_logvars=episode_latent_logvars[i].detach().cpu().numpy(),
                         episode_rewards=episode_rewards[i].detach().cpu().numpy(),
                         episode_returns=episode_returns[i].detach().cpu().numpy(),
                         episode_successes=np.array(episode_successes)[i],
                         )

        return episode_latent_means, episode_latent_logvars, \
               episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
               episode_returns, float(np.sum(episode_successes)>0)
