"""
this script uses the imu data from the real robot instead of quaternion value to train the tensegrity robot in simulation
"""

import copy
import time
from typing import Any, Optional, SupportsFloat
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque

# from rospkg import RosPack
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv
from tensegrity_sim import TensegrityEnv
from rospkg import RosPack

from EMAfilter import EMAFilter


def rescale_actions(low, high, action):
    """
    remapping the normalized actions from [-1, 1] to [low, high]
    """

    d = (high - low) / 2.0
    m = (high + low) / 2.0
    rescaled_action = action * d + m
    return rescaled_action


ADD_TENDON_LENGTH_OBSERVATION = True
ADD_ENC_VALUE_OBSERVATION = False
INITIALIZE_ROBOT_IN_AIR = False
PLOT_REWARD = False
INITIAL_TENSION = 0.0


class TensegrityEnvRealModelFullActuatorNoStiffnessImu(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100}
    info_keywords = ("rew_forward_x", "rew_ang_vel_pitch", "penalty_ang_vel_pitch")
    # TODO
    # 1.add action normalization
    # 2.add curriculum_learning
    # 3.use velocity cmd(angular)
    # 4.separate observation calculation function
    # 5.separate reward calculation function
    # 6.separate linear and angular momentum calculation functions
    # 7.add camera following
    # 8.consider terminate situation
    # 9.consider curriculum assistive force?
    # 10.add plot for debug in test mode

    def __init__(self, act_range, test, max_steps, resume=False, **kwargs):
        """
        resume training is abandoned due to mujoco supports evaluation along with training
        """

        # ema filter
        self.ema_filter = EMAFilter(0.2, np.array([0]*36))

        self.test = test
        self.is_params_set = False
        self.prev_action = None

        self.max_step = max_steps  # max_steps of one sub-env, used for curriculum learning
        self.resume = resume
        self.act_range = act_range  # tension force range
        print("act_range: ", self.act_range)

        self.max_episode = 2048  # maximum steps of every episode

        self.add_tendon_len_obs = ADD_TENDON_LENGTH_OBSERVATION
        self.add_enc_value_obs = ADD_ENC_VALUE_OBSERVATION
        self.plot_reward = PLOT_REWARD
        self.initial_tension = INITIAL_TENSION
        
        # control range
        self.num_actions = 24
        self.ctrl_max = np.array([0.] * self.num_actions)
        self.ctrl_min = np.array([-self.act_range] * self.num_actions)
        self.action_space_low = [-1.0] * self.num_actions
        self.action_space_high = [1.0] * self.num_actions

        # observation space
        num_obs_per_step = 63  # 36 + 24 + 24 + 3 = 87
        if self.add_tendon_len_obs:
            num_obs_per_step += 24
        if self.add_enc_value_obs:
            num_obs_per_step += 24
    
        self.n_obs_step = 1
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs_per_step*self.n_obs_step,))
        self.obs_deque = deque(maxlen=self.n_obs_step)
        for i in range(self.n_obs_step):
            self.obs_deque.appendleft(np.zeros(num_obs_per_step))

        self.check_steps = 200
        self.com_pos_deque = deque(maxlen=self.check_steps)
        for k in range(self.check_steps):
            self.com_pos_deque.appendleft(np.zeros(3))

        # velocity command
        self.vel_command = np.array([0., 0., 0.])  # linear velocity(x, y, z)

        self.actions = np.array([0.]*self.num_actions)

        self.episode_cnt = 0  # current episode step counter, used in test mode, reset to zero at the beginning of new episode
        self.step_cnt = 0  # never reset

        self.step_rate = 0.
        if self.test:
            self.step_rate = 1.0
            if self.plot_reward:
                self.draw_reward()

        self.rospack = RosPack()
        model_path = self.rospack.get_path('tensegrity_slam_sim') + '/models/scene_real_model_fullactuator_no_stiffness.xml'
        self.frame_skip = 2  # number of mujoco simulation steps per action step
        MujocoEnv.__init__(
            self,
            model_path,
            self.frame_skip,  # frame_skip
            observation_space=observation_space,
            **kwargs)

        utils.EzPickle.__init__(self)

        # robot initial state
        self.init_robot_in_air = INITIALIZE_ROBOT_IN_AIR # flag for reset the robot in air
        self.default_init_qpos = np.array([-0.125,  0.,  0.25,  1.,  0., 0.,  0.,
                                           0.125,  0.,  0.25, 1.,  0.,  0.,  0.,
                                           0., 0.125,  0.25,  0.70710678,  0.,  0.70710678, 0.,
                                           0., -0.125,  0.25,  0.70710678, 0.,  0.70710678,  0.,
                                           0.,  0., 0.375,  0.70710678,  0.70710678,  0.,  0.,
                                           0.,  0.,  0.125,  0.70710678,  0.70710678, 0.,  0.])
        self.default_init_qvel = np.array([0.0]*36)
        self.body_inertial = [np.diag(self.model.body_inertia[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                                                                "link{}".format(i+1))]) for i in range(6)]

        self.prev_com_pos = np.array([0., 0., 0.])
        self.reset_model()

    def _set_render_param(self):
        if self.test:
            self.mujoco_renderer.viewer._render_every_frame = False

    def _set_action_space(self):
        """
        always use normalized action space
        Noting: during env.step(), please rescale the action to the actual range!
        """
        low = np.asarray(self.action_space_low)
        high = np.asarray(self.action_space_high)
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        return self.action_space

    def _get_current_obs(self, imu_data, ten_length, actions, commands):
        """
        obs = imu + ten_len + actions + commands
        """
        return np.concatenate((imu_data, ten_length, actions.flatten(), commands))

    def _get_current_obs2(self, imu_data, ten_length, actions, commands):
        """
        obs = imu + imu_vel + ten_len + actions + commands
        """
        return np.concatenate((imu_data, ten_length, actions.flatten(), commands))

    def _get_stack_obs(self):
        return np.concatenate([self.obs_deque[i] for i in range(self.n_obs_step)])
    
    def step(self, action):
        """
        what we need do inside the step():
            - rescale_actions and add assistive force if needed---> TODO
            - mujoco simulation step forward
            - update flag and counters(such as step_cnt)
            - calculate the observations
            - calculate reward
            - check terminate conditions and truncated condition separately(timeout): reference->https://github.com/openai/gym/issues/2510
            - return
        action: (24,) normalized actions[-1,1] directly from policy
        """
        if not self.is_params_set:
            self._set_render_param()
            self.is_params_set = True
        
        if self.prev_action is None:
            self.prev_action = action

        # rescale action to tension force first
        tension_force = rescale_actions(self.ctrl_min, self.ctrl_max, action)

        # add external disturbance to center of each rod--> [N]
        self.data.qfrc_applied[:] = 0.02 * self.step_rate * np.random.randn(len(self.data.qfrc_applied))

        # add external assistive force curriculum
        self.data.xfrc_applied[:] = 0.0
        
        # add action(tension force) noise from [0.95, 1.05]--> percentage
        tension_force *= np.random.uniform(0.98, 1.02, self.num_actions)
        average_tension_force = np.mean(tension_force)
        # do simulation
        self._step_mujoco_simulation(tension_force, self.frame_skip)  # self.frame_skip=2, mujoco_step=200hz [0.95, 1.05]

        # update flags
        self.episode_cnt += 1
        self.step_cnt += 1

        # calculate the observations and update the observation deque
        if self.add_tendon_len_obs:
            imu_data = self.ema_filter.update(self.data.sensordata)
            tendon_length = self.data.ten_length
            cur_step_obs = self._get_current_obs(imu_data, tendon_length, action, self.vel_command)
        else:
            cur_step_obs = self._get_current_obs_2(imu_data, tendon_length, action, self.vel_command)
        self.obs_deque.appendleft(cur_step_obs)
        obs = self._get_stack_obs()

        # calculate the rewards
        current_com_pos = np.mean(copy.deepcopy(self.data.qpos.reshape(-1, 7)[:, 0:3]), axis=0)  # (3,)
        current_com_vel = np.mean(copy.deepcopy(self.data.qvel.reshape(-1, 6)), axis=0)  # (6,)
        current_ang_momentum = self.calculate_angular_momentum(self.data.qpos,
                                                               self.data.qvel,
                                                               current_com_pos,
                                                                current_com_vel[0:3])

        self.prev_com_pos = current_com_pos

        if np.dot(current_com_vel[0:2], self.vel_command[0:2]) > np.linalg.norm(self.vel_command)**2: # if v_x * v_x_cmd > ||v_x_cmd||^2
            self.velocity_reward = 1.0
        else:
            # velocity_reward = e^(-12*(v_x - v_x_cmd)^2)
            self.velocity_reward = np.exp(-6.0*(np.dot(current_com_vel[0:2], self.vel_command[0:2]) - np.linalg.norm(self.vel_command[0:2])**2)**2)
            #self.velocity_reward = np.exp(-10.0*(np.dot(current_com_vel[0:2], self.vel_command[0:2]) - np.linalg.norm(self.vel_command[0:2]))**2)
            #self.velocity_reward = np.exp(-8.0*np.square(current_com_vel[0:2] - self.vel_command[0:2]).sum())
        self.ang_momentum_penalty = current_ang_momentum[1] * int(current_ang_momentum[1] < 0.)
        self.ang_momentum_reward = current_ang_momentum[1] * int(current_ang_momentum[1] > 0.)
        
        self.action_penalty = -0.000 * np.linalg.norm(action) * self.step_rate # pre 0.001
        self.contorl_penalty = -0.000 * np.linalg.norm(action - self.prev_action) * self.step_rate
        #self.current_step_total_reward = self.velocity_reward + 1.5 * self.ang_momentum_reward + 5.0 * self.ang_momentum_penalty + self.action_penalty + self.contorl_penalty
        self.current_step_total_reward = self.velocity_reward + 1.5 * self.ang_momentum_reward + 5.0 * self.ang_momentum_penalty + self.action_penalty + self.contorl_penalty
        
        ## update prev_action
        self.prev_action = action

        rew_dict = {}
        if self.test:
            """
            print("------------", self.episode_cnt)
            # print("tendon_length", self.data.ten_length)
            print("x distance", current_com_pos[0])
            #print("angular momentum pitch", current_ang_momentum[1])
            print("current reward", self.current_step_total_reward)
            print("forward_x_reward", self.forward_x_reward)
            """
            #print("current reward", self.current_step_total_reward)
            print("velocity_reward", self.velocity_reward)
            #print("current_velocity", current_com_vel[0:2])
            #print("angular momentum pitch", current_ang_momentum[1])
            #print("ang_momentum_reward", self.ang_momentum_reward)
            #print("ang_momentum_penalty", self.ang_momentum_penalty)
            #print("current force", average_tension_force)
            #print("tension force", tension_force)
            rew_dict = {
                "ang_momentum_reward": self.ang_momentum_reward,
                "angular_momentum_penalty": self.ang_momentum_penalty
            }
            #self.fig1.canvas.draw()
            #self.fig1.canvas.flush_events()
            if self.plot_reward:
                self.fig2.canvas.draw()
                self.fig2.canvas.flush_events()
            #print("actutor velocity", self.data.actuator_velocity[0:3])
            #print("tendon velocity", self.data.ten_velocity[0:3])
            #print("diff velocity", self.data.ten_velocity[0:3]/1.0 - self.data.actuator_velocity[0:3])
                
            # print(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
            # ground_contact_position = []
            # for i in range(self.data.ncon):
            #     con = self.data.contact[i]
            #     print(f'contact{i}', con.geom1, con.geom2)
            #     if con.geom1 == 0 or con.geom2 == 0:
            #         print(con.pos)
        #
        # ang_pitch_vel_reward = np.exp(-4.0*np.abs(self.vel_command[1] - current_com_vel[4]))
        # ang_pitch_vel_penalty = current_com_vel[4] * int((current_com_vel[4] < 0.))
        # # current_step_total_reward = 0.25*forward_x_reward + 1.5*ang_pitch_vel_reward + 0.75*ang_pitch_vel_penalty
        # current_step_total_reward = 1.00 * forward_x_reward + 0.0 * ang_pitch_vel_reward + 0.0 * ang_pitch_vel_penalty
        # rew_dict = {
        #     "rew": current_step_total_reward,
        #     "rew_forward_x": forward_x_reward,
        #     "rew_ang_vel_pitch": ang_pitch_vel_reward,
        #     "penalty_ang_vel_pitch": ang_pitch_vel_penalty
        # }

        # check terminate and truncated
        self.com_pos_deque.appendleft(current_com_pos)
        terminated = False
        if self.episode_cnt > 400:
            terminated = np.linalg.norm(self.com_pos_deque[0] - self.com_pos_deque[-1]) < 0.03
        if terminated:
            self.current_step_total_reward += -5.0

        truncated = not (self.episode_cnt < self.max_episode)

        return (
            obs,
            self.current_step_total_reward,
            terminated,
            truncated,
            rew_dict
        )

    def reset_model(self):

        self.episode_cnt = 0

        # update step_rate and max_episode value at the beginning of every episode
        if self.max_step is not None:  # training or resume training mode
            self.step_rate = min(float(self.step_cnt) / self.max_step, 1)
        # self.max_episode = 512 + 1024 * self.step_rate

        # sample random initial pose
        qpos_addition = np.random.uniform(-0.00, 0.00, len(self.default_init_qpos)) * self.step_rate  # TODO:BUG

        qpos = self.default_init_qpos + qpos_addition
        if (self.init_robot_in_air and self.step_rate > 0.2) or self.test:
            qpos += np.array([0, 0, 1, 0, 0, 0, 0,
                              0, 0, 1, 0, 0, 0, 0,
                              0, 0, 1, 0, 0, 0, 0,
                              0, 0, 1, 0, 0, 0, 0,
                              0, 0, 1, 0, 0, 0, 0,
                              0, 0, 1, 0, 0, 0, 0
                              ]) * np.random.uniform(0.00, 0.00)

        # sample random initial vel
        qvel_addition = np.random.uniform(-0.0, 0.0, len(self.default_init_qvel)) * self.step_rate
        qvel = self.default_init_qvel + qvel_addition

        self.set_state(qpos, qvel)  # reset the values of mujoco model(robot)

        # switch to new command
        if self.test:
            self.vel_command = [0.8, 0.0, 0.0]
        else:
            #v = np.random.uniform(0.4, 0.7)
            v = 0.8
            self.vel_command = [v, 0.0, 0.0]

        # calculate the current step observations and fill out the observation buffer
        zero_actions = np.array([0.]*self.num_actions)
        
        if self.add_tendon_len_obs:
            imu_data = self.ema_filter.update(self.data.sensordata)
            tendon_length = self.data.ten_length
            cur_step_obs = self._get_current_obs(imu_data, tendon_length, zero_actions, self.vel_command)
        else:
            cur_step_obs = self._get_current_obs_2(imu_data, tendon_length, zero_actions, self.vel_command)
        for i in range(self.n_obs_step):
            self.obs_deque.appendleft(cur_step_obs)
        # update the com state
        self.prev_com_pos = np.mean(copy.deepcopy(self.data.qpos.reshape(-1, 7)[:, 0:3]), axis=0)  # (3,)
        for k in range(self.check_steps):
            self.com_pos_deque.appendleft(self.prev_com_pos)
        
        # add initial tension
        self.data.ctrl[:] = self.initial_tension

        # return the stacked obs as the initial obs of episode
        return self._get_stack_obs()

    def calculate_angular_momentum(self, qpos, qvel, com_position, com_vel):
        total_angular_momentum = np.zeros(3)
        body_mass = 0.65
        links_position = qpos.reshape(-1, 7)[:, 0:3]
        links_velocity = qvel.reshape(-1, 6)[:, 0:3]
        links_ang_vel = qvel.reshape(-1, 6)[:, 3:]
        for i in range(6):
            rot_mat = self.data.xmat[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link{}".format(i+1))].reshape(3, 3)  # R_i: rotation matrix
            angular_momentum = body_mass*np.cross((links_position[i] - com_position), (links_velocity[i] - com_vel))
            angular_momentum += rot_mat @ self.body_inertial[i] @ rot_mat.transpose() @ links_ang_vel[i]

            total_angular_momentum += angular_momentum

        return total_angular_momentum
