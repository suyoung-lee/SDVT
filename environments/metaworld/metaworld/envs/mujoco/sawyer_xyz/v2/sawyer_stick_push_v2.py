import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerStickPushEnvV2(SawyerXYZEnv):
    def __init__(self):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.08, 0.58, 0.000)
        obj_high = (-0.03, 0.62, 0.001)
        goal_low = (0.399, 0.55, 0.1319)
        goal_high = (0.401, 0.6, 0.1321)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'stick_init_pos': np.array([-0.1, 0.6, 0.02]),
            'hand_init_pos': np.array([0, 0.6, 0.2]),
        }
        self.goal = self.init_config['stick_init_pos']
        self.stick_init_pos = self.init_config['stick_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        # For now, fix the object initial position.
        self.obj_init_pos = np.array([0.2, 0.6, 0.0])
        self.obj_init_qpos = np.array([0.0, 0.0])
        self.obj_space = Box(np.array(obj_low), np.array(obj_high))
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_stick_obj.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        stick = obs[4:7]
        container = obs[11:14]
        reward, tcp_to_obj, tcp_open, container_to_target, grasp_reward, stick_in_place = self.compute_reward(action, obs)
        success = float(np.linalg.norm(container - self._target_pos) <= 0.12)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(self.touching_object and (tcp_open > 0) and (stick[2] - 0.01 > self.stick_init_pos[2]))

        info = {
            'success': grasp_success and success,
            'near_object': near_object,
            'grasp_success': grasp_success,
            'grasp_reward': grasp_reward,
            'in_place_reward': stick_in_place,
            'obj_to_target': container_to_target,
            'unscaled_reward': reward,

        }

        return reward, info

    def _get_pos_objects(self):
        return np.hstack((
            self.get_body_com('stick').copy(),
            self._get_site_pos('insertion') + np.array([.0, .09, .0]),
        ))

    def _get_quat_objects(self):
        return np.hstack((
            Rotation.from_matrix(self.data.get_body_xmat('stick')).as_quat(),
            np.array([0.,0.,0.,0.])
        ))

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict['state_achieved_goal'] = self._get_site_pos(
            'insertion'
        ) + np.array([.0, .09, .0])
        return obs_dict

    def _set_stick_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[16:18] = pos.copy()
        qvel[16:18] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        self.stick_init_pos = self.init_config['stick_init_pos']
        self._target_pos = np.array([0.4, 0.6, self.stick_init_pos[-1]])

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
                goal_pos = self._get_state_rand_vec()
            self.stick_init_pos = np.concatenate((goal_pos[:2], [self.stick_init_pos[-1]]))
            self._target_pos = np.concatenate((goal_pos[-3:-1], [self._get_site_pos('insertion')[-1]]))

        self._set_stick_xyz(self.stick_init_pos)
        self._set_obj_xyz(self.obj_init_qpos)
        self.obj_init_pos = self.get_body_com('object').copy()

        return self._get_obs()
    
    def _gripper_caging_reward(self,
                               action,
                               obj_pos,
                               obj_radius,
                               pad_success_thresh,
                               object_reach_radius,
                               xz_thresh,
                               desired_gripper_effort=1.0,
                               high_density=False,
                               medium_density=False):
        """Reward for agent grasping obj
            Args:
                action(np.ndarray): (4,) array representing the action
                    delta(x), delta(y), delta(z), gripper_effort
                obj_pos(np.ndarray): (3,) array representing the obj x,y,z
                obj_radius(float):radius of object's bounding sphere
                pad_success_thresh(float): successful distance of gripper_pad
                    to object
                object_reach_radius(float): successful distance of gripper center
                    to the object.
                xz_thresh(float): successful distance of gripper in x_z axis to the
                    object. Y axis not included since the caging function handles
                        successful grasping in the Y axis.
        """
        if high_density and medium_density:
            raise ValueError("Can only be either high_density or medium_density")
        # MARK: Left-right gripper information for caging reward----------------
        left_pad = self.get_body_com('leftpad')
        right_pad = self.get_body_com('rightpad')

        # get current positions of left and right pads (Y axis)
        pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
        # compare *current* pad positions with *current* obj position (Y axis)
        pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
        # compare *current* pad positions with *initial* obj position (Y axis)
        pad_to_objinit_lr = np.abs(pad_y_lr - self.stick_init_pos[1])

        caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
        caging_lr = [reward_utils.tolerance(
            pad_to_obj_lr[i],  # "x" in the description above
            bounds=(obj_radius, pad_success_thresh),
            margin=caging_lr_margin[i],  # "margin" in the description above
            sigmoid='long_tail',
        ) for i in range(2)]
        caging_y = reward_utils.hamacher_product(*caging_lr)

        # MARK: X-Z gripper information for caging reward-----------------------
        tcp = self.tcp_center
        xz = [0, 2]

        caging_xz_margin = np.linalg.norm(self.stick_init_pos[xz] - self.init_tcp[xz])
        caging_xz_margin -= xz_thresh
        caging_xz = reward_utils.tolerance(
            np.linalg.norm(tcp[xz] - obj_pos[xz]),  # "x" in the description above
            bounds=(0, xz_thresh),
            margin=caging_xz_margin,  # "margin" in the description above
            sigmoid='long_tail',
        )

        # MARK: Closed-extent gripper information for caging reward-------------
        gripper_closed = min(max(0, action[-1]), desired_gripper_effort) \
                         / desired_gripper_effort

        # MARK: Combine components----------------------------------------------
        caging = reward_utils.hamacher_product(caging_y, caging_xz)
        gripping = gripper_closed if caging > 0.97 else 0.
        caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

        if high_density:
            caging_and_gripping = (caging_and_gripping + caging) / 2
        if medium_density:
            tcp = self.tcp_center
            tcp_to_obj = np.linalg.norm(obj_pos - tcp)
            tcp_to_obj_init = np.linalg.norm(self.stick_init_pos - self.init_tcp)
            reach_margin = abs(tcp_to_obj_init - object_reach_radius)
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, object_reach_radius),
                margin=reach_margin,
                sigmoid='long_tail',
            )
            caging_and_gripping = (caging_and_gripping + reach) / 2

        return caging_and_gripping

    def compute_reward(self, action, obs):
        _TARGET_RADIUS = 0.12
        tcp = self.tcp_center
        stick = obs[4:7] + np.array([.015, .0, .0])
        container = obs[11:14]
        tcp_opened = obs[3]
        target = self._target_pos

        tcp_to_stick = np.linalg.norm(stick - tcp)
        stick_to_target = np.linalg.norm(stick - target)
        stick_in_place_margin = (np.linalg.norm(self.stick_init_pos - target)) - _TARGET_RADIUS
        stick_in_place = reward_utils.tolerance(stick_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=stick_in_place_margin,
                                    sigmoid='long_tail',)

        container_to_target = np.linalg.norm(container - target)
        container_in_place_margin = np.linalg.norm(self.obj_init_pos - target) - _TARGET_RADIUS
        container_in_place = reward_utils.tolerance(container_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=container_in_place_margin,
                                    sigmoid='long_tail',)

        object_grasped = self._gripper_caging_reward(
            action=action,
            obj_pos=stick,
            obj_radius=0.04,
            pad_success_thresh=0.05,
            object_reach_radius=0.01,
            xz_thresh=0.01,
            high_density=True
        )

        reward = object_grasped

        if tcp_to_stick < 0.02 and (tcp_opened > 0) and \
                (stick[2] - 0.01 > self.stick_init_pos[2]):
            object_grasped = 1
            reward = 2. + 5. * stick_in_place + 3. * container_in_place

            if container_to_target <= _TARGET_RADIUS:
                reward = 10.

        return [reward, tcp_to_stick, tcp_opened, container_to_target, object_grasped, stick_in_place]
