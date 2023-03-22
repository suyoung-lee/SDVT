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


def semi_circle_goal_sampler():
    r = 1.0
    angle = random.uniform(0, np.pi)
    goal = r * np.array((np.cos(angle), np.sin(angle)))
    return goal


def circle_goal_sampler():
    r = 1.0
    angle = random.uniform(0, 2*np.pi)
    goal = r * np.array((np.cos(angle), np.sin(angle)))
    return goal


GOAL_SAMPLERS = {
    'semi-circle': semi_circle_goal_sampler,
    'circle': circle_goal_sampler,
}


class ML10(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, max_episode_steps=100, goal_sampler=None):
        if callable(goal_sampler):
            self.goal_sampler = goal_sampler
        elif isinstance(goal_sampler, str):
            self.goal_sampler = GOAL_SAMPLERS[goal_sampler]
        elif goal_sampler is None:
            self.goal_sampler = semi_circle_goal_sampler
        else:
            raise NotImplementedError(goal_sampler)

        ml10 = metaworld.ML10()
        self.train_env_name_list = [name for name,_ in ml10.train_classes.items()]
        self.train_env_cls_list = [env_cls() for _,env_cls in ml10.train_classes.items()]
        self.test_env_name_list = [name for name,_ in ml10.test_classes.items()]
        self.test_envcls_list = [env_cls() for _,env_cls in ml10.test_classes.items()]
        self.train_tasks = ml10.train_tasks
        self.test_tasks = ml10.test_tasks


        self.reset_task()
        print('reset task called')
        self.task_dim = 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        # we convert the actions from [-1, 1] to [-0.1, 0.1] in the step() function
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self._max_episode_steps = max_episode_steps


    def sample_task(self):
        goal = self.goal_sampler()
        return goal

    def set_task(self, task):
        self._goal = task

    def get_task(self):
        return self._goal

    def reset_task(self, task=None, testing=False):
        if task is None:
            task = self.sample_task()

        #ml 10
            if not testing:
                env_ind = random.choice(range(10))
                self._env = self.train_env_cls_list[env_ind]
                _env_name = self.train_env_name_list[env_ind]
                _task = random.choice([_task for _task in self.train_tasks
                                      if _task.env_name == _env_name])

                self._env.set_task(_task)

        print('1', self._env)
        self.set_task(task)

        return task

    def reset_model(self):
        self._state = np.zeros(2)

        self._state_ml10 = self._env.reset()
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):

        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action), action

        self._state = self._state + 0.1 * action
        reward = - np.linalg.norm(self._state - self._goal, ord=2)
        done = False
        ob = self._get_obs()
        info = {'task': self.get_task()}

        a = self._env.action_space.sample()
        obs_ml10, reward_ml10, done_ml10, info_ml10 = self._env.step(a)
        print('2', obs_ml10, reward_ml10, done_ml10, info_ml10)



        return ob, reward, done, info
