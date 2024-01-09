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


def rescale_actions(low, high, action):
    """
    remapping the normalized actions from [-1, 1] to [low, high]
    """

    d = (high - low) / 2.0
    m = (high + low) / 2.0
    rescaled_action = action * d + m
    return rescaled_action


USE_ANG_VEL_OBS = False
USE_ONLY_TENDON_LENGTH_OBS = True
ADD_ASSISTIVE_FORCE = False
ADD_TENDON_LENGTH_OBSERVATION = False
ADD_TENDON_VEL_OBSERVATION = False
INITIALIZE_ROBOT_IN_AIR = False
PLOT_REWARD = False
PLOT_SENSOR = False
INITIAL_TENSION = 0.0


class TensegrityEnvRealModelFullActuatorNoStiffness(MujocoEnv, utils.EzPickle):
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
        self.plot_sensor = PLOT_SENSOR
        if self.plot_sensor:
            # sensor data
            self.sensor_data = []

        self.test = test
        self.is_params_set = False
        self.prev_action = None

        self.max_step = max_steps  # max_steps of one sub-env, used for curriculum learning
        self.resume = resume
        self.act_range = act_range  # tension force range
        print("act_range: ", self.act_range)

        self.max_episode = 2048  # maximum steps of every episode

        self.add_assistive_force = ADD_ASSISTIVE_FORCE
        self.add_tendon_len_obs = ADD_TENDON_LENGTH_OBSERVATION
        self.plot_reward = PLOT_REWARD
        self.add_tendon_vel_obs = ADD_TENDON_VEL_OBSERVATION
        self.initial_tension = INITIAL_TENSION
        
        # control range
        self.num_actions = 24
        self.ctrl_max = np.array([0.] * self.num_actions)
        self.ctrl_min = np.array([-self.act_range] * self.num_actions)
        self.action_space_low = [-1.0] * self.num_actions
        self.action_space_high = [1.0] * self.num_actions

        # observation space
        self.use_ang_vel_obs = USE_ANG_VEL_OBS
        self.use_only_tendon_len_obs = USE_ONLY_TENDON_LENGTH_OBS
        num_obs_per_step = 51  # 24 + 24 + 3 = 51  or 24 + 18 + 24 + 3 = 69
        if self.use_only_tendon_len_obs:
            num_obs_per_step += 24
        if self.use_ang_vel_obs:
            num_obs_per_step += 18
        if self.add_tendon_len_obs:
            num_obs_per_step += 24
        if self.add_tendon_vel_obs:
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
            self.step_rate = 0.0
            if self.plot_reward:
                self.draw_reward()

        self.rospack = RosPack()
        model_path = self.rospack.get_path('tensegrity_slam_sim') + '/models/scene_real_model_fullactuator_no_stiffness.xml'
        #model_path = self.rospack.get_path('tensegrity_slam_sim') + '/models/scene_rough_terrain.xml'
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

    def _get_current_obs(self, qpos, qvel, actions, commands):
        """
        calculate one step observations
        """
        body_quat = qpos.reshape((-1, 7))[:, 3:]  # mujoco uses scaler-first quaternion [w, x, y, z]
        body_ang_vel_world = qvel.reshape((-1, 6))[:, 3:]

        body_quat_w_last = np.zeros_like(body_quat)
        body_quat_w_last[:, -1] = body_quat[:, 0]
        body_quat_w_last[:, 0:3] = body_quat[:, 1:]
        rot_matrix = R.from_quat(body_quat_w_last).as_matrix()  # stack of rotation matrix

        body_ang_vel_local = np.zeros_like(body_ang_vel_world)
        for i in range(body_ang_vel_local.shape[0]):
            body_ang_vel_local[i] = np.dot(rot_matrix[i].transpose(), body_ang_vel_world[i])

        return np.concatenate((body_quat.flatten(), body_ang_vel_local.flatten(), actions.flatten(), commands))

    def _get_current_obs2(self, qpos, qvel, actions, commands, tendon_length):
        """
        calculate one step observations
        """
        body_quat = qpos.reshape((-1, 7))[:, 3:]  # mujoco uses scaler-first quaternion [w, x, y, z]
        body_ang_vel_world = qvel.reshape((-1, 6))[:, 3:]

        body_quat_w_last = np.zeros_like(body_quat)
        body_quat_w_last[:, -1] = body_quat[:, 0]
        body_quat_w_last[:, 0:3] = body_quat[:, 1:]
        rot_matrix = R.from_quat(body_quat_w_last).as_matrix()  # stack of rotation matrix

        body_ang_vel_local = np.zeros_like(body_ang_vel_world)
        for i in range(body_ang_vel_local.shape[0]):
            body_ang_vel_local[i] = np.dot(rot_matrix[i].transpose(), body_ang_vel_world[i])

        return np.concatenate((body_quat.flatten(), body_ang_vel_local.flatten(), tendon_length, actions.flatten(), commands))
    
    def _get_current_obs3(self, qpos, qvel, actions, commands, tendon_length, tendon_velocity):
        """
        quaternion + tendon_length + actions + commands + tendon_velocity
        """
        return np.concatenate([self._get_current_obs2(qpos, qvel, actions, commands, tendon_length), tendon_velocity])

    def _get_current_obs_4(self, qpos, actions, commands):
        """
        quaternion + actions + commands
        """
        body_quat = qpos.reshape((-1, 7))[:, 3:]  # mujoco uses scaler-first quaternion [w, x, y, z]
        
        body_quat_w_last = np.zeros_like(body_quat)
        body_quat_w_last[:, -1] = body_quat[:, 0]
        body_quat_w_last[:, 0:3] = body_quat[:, 1:]

        return np.concatenate((body_quat.flatten(), actions.flatten(), commands))

    def _get_current_obs_5(self, qpos, actions, commands, tendon_length):
            """
            quaternion + tendon_length + actions + commands
            """
            body_quat = qpos.reshape((-1, 7))[:, 3:]  # mujoco uses scaler-first quaternion [w, x, y, z]
            
            body_quat_w_last = np.zeros_like(body_quat)
            body_quat_w_last[:, -1] = body_quat[:, 0]
            body_quat_w_last[:, 0:3] = body_quat[:, 1:]

            return np.concatenate((body_quat.flatten(), tendon_length, actions.flatten(), commands))

    def _get_stack_obs(self):
        return np.concatenate([self.obs_deque[i] for i in range(self.n_obs_step)])
    
    def draw_reward(self):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        self.fig1, self.ax1 = plt.subplots()
        self.xdata1, self.ydata1 = [], []
        self.ln1, = plt.plot([], [], 'r-', animated=True)
        def init1():
            self.ax1.set_xlim(0, 10000)
            self.ax1.set_ylim(-2, 2)
            self.ax1.set_ylabel("forward_x_reward")
            return self.ln1,
        def update1(frame):
            self.xdata1.append(frame)
            self.ydata1.append(self.forward_x_reward)
            self.ln1.set_data(self.xdata1, self.ydata1)
            return self.ln1,
        ani1 = FuncAnimation(self.fig1, update1, frames=np.linspace(0, 10000, 10000),  
                            init_func=init1, blit=True)
        
        self.fig2, self.ax2 = plt.subplots()
        self.xdata2, self.ydata2 = [], []
        self.ln2, = plt.plot([], [], 'r-', animated=True)
        def init2():
            self.ax2.set_xlim(0, 10000)
            self.ax2.set_ylim(-2, 2)
            self.ax2.set_ylabel("ang_momentum_reward")
            return self.ln2,
        def update2(frame):
            self.data2.append(frame)
            self.ydata2.append(self.ang_momentum_reward)
            self.ln2.set_data(self.xdata2, self.ydata2)
            return self.ln2,
        ani2 = FuncAnimation(self.fig2, update2, frames=np.linspace(0, 10000, 10000), interval=1,
                            init_func=init2, blit=True)
        
        self.fig3, self.ax3 = plt.subplots()
        self.xdata3, self.ydata3 = [], []
        self.ln3, = plt.plot([], [], 'r-', animated=True)
        def init3():
            self.ax3.set_xlim(0, 10000)
            self.ax3.set_ylim(-2, 2)
            self.ax3.set_ylabel("total_reward")
            return self.ln3,
        def update3(frame):
            self.data3.append(frame)
            self.ydata3.append(self.current_step_total_reward)
            self.ln3.set_data(self.xdata3, self.ydata3)
            return self.ln3,
        ani2 = FuncAnimation(self.fig3, update3, frames=np.linspace(0, 10000, 10000), interval=1,
                            init_func=init3, blit=True)
        
        plt.show(block=False)
        
    def plot_sensor_data(self):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self.sensor_data)
        plt.title("Acceleration Sensor Data")
        plt.xlabel("Step")
        plt.ylabel("link1_acceleration_x")
        plt.show()
        
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
        if self.add_assistive_force and self.episode_cnt > 300:
            if np.linalg.norm(self.com_pos_deque[0] - self.com_pos_deque[50]) < 0.03:
                print("add assistive force", self.episode_cnt)
                body_name_list = ["link1", "link2", "link3", "link4", "link5", "link6"]
                body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name_list[i]) for i in range(len(body_name_list))]
                # TODO: check the validation
                x_frc = np.zeros((6, 6))
                x_frc[:, 0] = max((1.0-0), 0.) * 100.0
                self.data.xfrc_applied[body_ids] = x_frc

        # add action(tension force) noise from [0.95, 1.05]--> percentage
        tension_force *= np.random.uniform(0.98, 1.02, self.num_actions)
        average_tension_force = np.mean(tension_force)
        # do simulation
        self._step_mujoco_simulation(tension_force, self.frame_skip)  # self.frame_skip=2, mujoco_step=200hz [0.95, 1.05]

        # update flags
        self.episode_cnt += 1
        self.step_cnt += 1

        # calculate the observations and update the observation deque
        if self.use_only_tendon_len_obs:
            tendon_length = self.data.ten_length
            cur_step_obs = self._get_current_obs_5(self.data.qpos, action, self.vel_command, tendon_length)
        elif self.add_tendon_vel_obs:
            tendon_velocity = self.data.ten_velocity
            tendon_length = self.data.ten_length
            cur_step_obs = self._get_current_obs3(self.data.qpos, self.data.qvel, action, self.vel_command, tendon_length, tendon_velocity)
        elif self.add_tendon_len_obs:
            tendon_length = self.data.ten_length
            cur_step_obs = self._get_current_obs2(self.data.qpos, self.data.qvel, action, self.vel_command, tendon_length)
        elif self.use_ang_vel_obs:
            cur_step_obs = self._get_current_obs(self.data.qpos, self.data.qvel, action, self.vel_command)
        else:
            cur_step_obs = self._get_current_obs_4(self.data.qpos, action, self.vel_command)
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
        
        # save sensor 
        if self.plot_sensor:
            self.sensor_data.append(self.data.sensordata[0])
        if terminated or truncated:
            if self.plot_sensor:
                # self.sensor_data = np.array(self.sensor_data)
                # self.sensor_data = self.sensor_data.reshape(-1, 6)
                # np.save("sensor_data.npy", self.sensor_data)
                # print("sensor data saved!")
                self.plot_sensor_data()

        return (
            obs,
            self.current_step_total_reward,
            terminated,
            truncated,
            rew_dict
        )

    def reset_model(self):
        
        if self.plot_sensor:
            self.sensor_data.clear()

        self.episode_cnt = 0

        # update step_rate and max_episode value at the beginning of every episode
        if self.max_step is not None:  # training or resume training mode
            self.step_rate = min(float(self.step_cnt) / self.max_step, 1)
        # self.max_episode = 512 + 1024 * self.step_rate

        # sample random initial pose
        # qpos_addition = np.random.uniform(-0.00, 0.00, len(self.default_init_qpos)) * self.step_rate  # TODO:BUG
        # initial tendon length
        self.data.ten_length[:] = [0.30]*24
        qpos_addition = np.random.uniform(-0.02, 0.02, len(self.default_init_qpos)) * self.step_rate  # TODO:BUG

        qpos = np.array([0.14717668,  0.14711882,  0.15701801,  0.86432397, -0.40548401,  0.2194443,
                        -0.20092532,  0.350647,    0.11930152,  0.06542414,  0.79409071, -0.2563381,
                        0.54759233,  0.06207542,  0.22993135,  0.20179415,  0.06074503,  0.50408641,
                        -0.1424163,   0.77721656, -0.34863865,  0.27766309,  0.00355943,  0.15893443,
                        0.39771177, -0.1131317,   0.89793995, -0.1507661,   0.35460463,  0.19562937,
                        0.15674695,  0.86554097,  0.36059424,  0.23697669, -0.25426887,  0.19753333,
                        0.03760321,  0.07496286,  0.74249165,  0.51360453,  0.28749698, -0.31978435,
                    ])
        qpos += qpos_addition
    
        # qpos = self.default_init_qpos + qpos_addition
        # if (self.init_robot_in_air and self.step_rate > 0.2) or self.test:
        #     qpos += np.array([0, 0, 1, 0, 0, 0, 0,
        #                       0, 0, 1, 0, 0, 0, 0,
        #                       0, 0, 1, 0, 0, 0, 0,
        #                       0, 0, 1, 0, 0, 0, 0,
        #                       0, 0, 1, 0, 0, 0, 0,
        #                       0, 0, 1, 0, 0, 0, 0
        #                       ]) * np.random.uniform(0.00, 0.00)

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
        
        if self.use_only_tendon_len_obs:
            tendon_length = self.data.ten_length
            cur_step_obs = self._get_current_obs_5(self.data.qpos, zero_actions, self.vel_command, tendon_length)
        elif self.add_tendon_vel_obs:
            tendon_velocity = self.data.ten_velocity
            tendon_length = self.data.ten_length
            cur_step_obs = self._get_current_obs3(self.data.qpos, self.data.qvel, zero_actions, self.vel_command, tendon_length, tendon_velocity)
        elif self.add_tendon_len_obs:
            tendon_length = self.data.ten_length
            cur_step_obs = self._get_current_obs2(self.data.qpos, self.data.qvel, zero_actions, self.vel_command, tendon_length)
        elif self.use_ang_vel_obs:
            cur_step_obs = self._get_current_obs(self.data.qpos, self.data.qvel, zero_actions, self.vel_command)
        else:
            cur_step_obs = self._get_current_obs_4(self.data.qpos, zero_actions, self.vel_command)
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
