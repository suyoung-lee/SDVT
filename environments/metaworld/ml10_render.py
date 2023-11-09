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

class ML10_RENDER(Env):

    def __init__(self, max_episode_steps=500):

        ml10 = metaworld.ML10()
        self.train_env_name_list = [name for name,_ in ml10.train_classes.items()]
        self.train_env_cls_list = [env_cls() for _,env_cls in ml10.train_classes.items()]
        self.test_env_name_list = [name for name,_ in ml10.test_classes.items()]
        self.test_env_cls_list = [env_cls() for _,env_cls in ml10.test_classes.items()]
        self.train_tasks = ml10.train_tasks
        self.test_tasks = ml10.test_tasks

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
            env_ind = random.choice(range(10))
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
            if env_ind <10:
                self._env = self.train_env_cls_list[env_ind]
                _env_name = self.train_env_name_list[env_ind]
                self._env.set_task([_task for _task in self.train_tasks
                                      if _task.env_name == _env_name][subtask_ind])
            else:
                self._env = self.test_env_cls_list[env_ind-10]
                _env_name = self.test_env_name_list[env_ind-10]
                self._env.set_task([_task for _task in self.test_tasks
                                      if _task.env_name == _env_name][subtask_ind])
        x= self.reset()
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

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._state, reward, done, info = self._env.step(action)
        ob = self._get_obs()

        info = {'task': self.get_task(), 'success': info['success'], 'image': self._env.render(offscreen=True)}

        return ob, reward, done, info
