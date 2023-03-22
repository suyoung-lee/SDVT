import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerBinPickingEnvV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was often unsolvable because the cube could be located outside of
        the starting bin. It could even be near the base of the Sawyer and out
        of reach of the gripper. V2 changes the `obj_low` and `obj_high` bounds
        to fix this.
    Changelog from V1 to V2:
        - (7/20/20) Changed object initialization space
        - (7/24/20) Added Byron's XML changes
        - (11/23/20) Updated reward function to new pick-place style
    """

    def __init__(self):

        hand_low = (-0.5, 0.40, 0.07)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.21, 0.65, 0.02)
        obj_high = (-0.03, 0.75, 0.02)
        # Small bounds around the center of the target bin
        goal_low = np.array([0.1199, 0.699, -0.001])
        goal_high = np.array([0.1201, 0.701, +0.001])

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([-0.12, 0.7, 0.02]),
            'hand_init_pos': np.array((0, 0.6, 0.2)),
        }
        self.goal = np.array([0.12, 0.7, 0.02])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._target_to_obj_init = None

        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        )

        self.goal_and_obj_space = Box(
            np.hstack((goal_low[:2], obj_low[:2])),
            np.hstack((goal_high[:2], obj_high[:2])),
        )

        self.goal_space = Box(goal_low, goal_high)
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_bin_picking.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            near_object,
            grasp_success,
            obj_to_target,
            grasp_reward,
            in_place_reward
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(obj_to_target <= 0.05),
            'near_object': float(near_object),
            'grasp_success': float(grasp_success),
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place_reward,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return []

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('objGeom')

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('obj')

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        obj_height = self.get_body_com('obj')[2]

        if self.random_init:
            self.obj_init_pos = self._get_state_rand_vec()[:2]
            self.obj_init_pos = np.concatenate((self.obj_init_pos, [obj_height]))

        self._set_obj_xyz(self.obj_init_pos)
        self._target_pos = self.get_body_com('bin_goal')
        self._target_to_obj_init = None

        return self._get_obs()

    def compute_reward(self, action, obs):
        hand = obs[:3]
        obj = obs[4:7]

        target_to_obj = np.linalg.norm(obj - self._target_pos)
        if self._target_to_obj_init is None:
            self._target_to_obj_init = target_to_obj

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=self._target_to_obj_init,
            sigmoid='long_tail',
        )

        threshold = 0.03
        radii = [
            np.linalg.norm(hand[:2] - self.obj_init_pos[:2]),
            np.linalg.norm(hand[:2] - self._target_pos[:2])
        ]
        # floor is a *pair* of 3D funnels centered on (1) the object's initial
        # position and (2) the desired final position
        floor = min([
            0.02 * np.log(radius - threshold) + 0.2
            if radius > threshold else 0.0
            for radius in radii
        ])
        # prevent the hand from running into the edge of the bins by keeping
        # it above the "floor"
        above_floor = 1.0 if hand[2] >= floor else reward_utils.tolerance(
            max(floor - hand[2], 0.0),
            bounds=(0.0, 0.01),
            margin=0.05,
            sigmoid='long_tail',
        )

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            obj_radius=0.015,
            pad_success_thresh=0.05,
            object_reach_radius=0.01,
            xz_thresh=0.01,
            desired_gripper_effort=0.7,
            high_density=True,
        )
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        near_object = np.linalg.norm(obj - hand) < 0.04
        pinched_without_obj = obs[3] < 0.43
        lifted = obj[2] - 0.02 > self.obj_init_pos[2]
        # Increase reward when properly grabbed obj
        grasp_success = near_object and lifted and not pinched_without_obj
        if grasp_success:
            reward += 1. + 5. * reward_utils.hamacher_product(
                above_floor, in_place
            )
        # Maximize reward on success
        if target_to_obj < self.TARGET_RADIUS:
            reward = 10.

        return (
            reward,
            near_object,
            grasp_success,
            target_to_obj,
            object_grasped,
            in_place
        )
