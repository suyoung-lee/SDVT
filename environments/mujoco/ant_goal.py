import random

import numpy as np
from numpy import linalg as LA

from environments.mujoco.ant import AntEnv


class AntGoalEnv(AntEnv):
    def __init__(self, max_episode_steps=200):
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 4

        self.eval_task_list = [[0.5,0.0],[0.0,0.5],[-0.5,0.0],[0.0,-0.5],  [1.75,0.0],[0.0,1.75],[-1.75,0.0],[0.0,-1.75], [2.75,0.0],[0.0,2.75],[-2.75,0.0],[0.0,-2.75]] #case 4

        super(AntGoalEnv, self).__init__()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal_pos))  # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()

        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            task=self.get_task()
        )

    def sample_tasks(self, num_tasks):
        oracle = False
        if oracle:
            r = 3 * np.array([random.random() for _ in range(num_tasks)]) ** 0.5
            a = np.array([random.random() for _ in range(num_tasks)]) * 2 * np.pi
        else:
            if random.random() < 4.0/15.0:
                r = np.array([random.random() for _ in range(num_tasks)]) ** 0.5
            else:
                r = np.array([random.random() * 2.75 + 6.25 for _ in range(num_tasks)]) ** 0.5

            a = np.array([random.random() for _ in range(num_tasks)]) * 2 * np.pi

        return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

    def set_task(self, task):
        self.goal_pos = task

    def get_task(self):
        return self.goal_pos

    def get_test_task_list(self):
        return self.eval_task_list

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])


class AntGoalOracleEnv(AntGoalEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.goal_pos,
        ])