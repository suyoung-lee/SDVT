import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerHandlePullEnvV2(SawyerXYZEnv):
    
    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.8, -0.001)
        obj_high = (0.1, 0.9, +0.001)
        goal_low = (-0.1, 0.55, 0.04)
        goal_high = (0.1, 0.70, 0.18)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.9, 0.0]),
            'hand_init_pos': np.array((0, 0.6, 0.2),),
        }
        self.goal = np.array([0, 0.8, 0.14])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_handle_press.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        obj = obs[4:7]

        (reward,
         tcp_to_obj,
         tcp_open,
         obj_to_target,
         grasp_reward,
         in_place_reward
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(obj_to_target <= self.TARGET_RADIUS),
            'near_object': float(tcp_to_obj <= 0.05),
            'grasp_success': float(
                (tcp_open > 0) and
                (obj[2] - 0.03 > self.obj_init_pos[2])
            ),
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place_reward,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return []

    def _get_pos_objects(self):
        return self._get_site_pos('handleRight')

    def _get_quat_objects(self):
        return np.zeros(4)

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        self.obj_init_pos = (self._get_state_rand_vec()
                             if self.random_init
                             else self.init_config['obj_init_pos'])

        self.sim.model.body_pos[self.model.body_name2id('box')] = self.obj_init_pos
        self._set_obj_xyz(-0.1)
        self._target_pos = self._get_site_pos('goalPull')

        return self._get_obs()

    def compute_reward(self, action, obs):
        obj = obs[4:7]
        # Force target to be slightly above basketball hoop
        target = self._target_pos.copy()

        target_to_obj = abs(target[2] - obj[2])
        target_to_obj_init = abs(target[2] - self.obj_init_pos[2])

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            pad_success_thresh=0.05,
            obj_radius=0.022,
            object_reach_radius=0.01,
            xz_thresh=0.01,
            high_density=True,
        )
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        tcp_opened = obs[3]
        tcp_to_obj = np.linalg.norm(obj - self.tcp_center)

        if tcp_to_obj < 0.035 and tcp_opened > 0 and \
                obj[2] - 0.01 > self.obj_init_pos[2]:
            reward += 1. + 5. * in_place
        if target_to_obj < self.TARGET_RADIUS:
            reward = 10.
        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place
        )
