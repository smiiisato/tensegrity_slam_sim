import copy
from typing import Any, Optional, SupportsFloat
import mujoco
import numpy as np
from rospkg import RosPack
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv
from tensegrity_sim import TensegrityEnv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class TensegrityEnvRealmodelFullactuatorAngularvelocity(TensegrityEnv):

    def __init__(self, act_range=18.0, test=False, ros=False, max_steps=None, resume=False, **kwargs):
        self.action_length = 24
        self.is_params_set = False
        self.test = test
        self.ros = ros
        self.max_step = max_steps
        self.step_rate_max_cnt = 20000000
        self.resume = resume
        self.act_range = act_range
        print("act_range: ", self.act_range)

        # control range
        self.ctrl_max = [0]*self.action_length
        self.ctrl_min = [-self.act_range]*self.action_length

        # initial command, direction +x
        self.command = [0.0, 0.5, 0.0]
        self.local_command = None

        # flag for randomizing initial position
        #self.randomize_position = (self.resume or self.test)
        self.randomize_position = False

        self.n_prev = 1
        self.max_episode = 1000
    
        self.current_body_xpos = None
        self.current_body_xquat = None
        self.com_qvel = None
        self.prev_body_xpos = None
        self.prev_body_xquat = None
        self.prev_action = None
        self.angular_velocity = None

        self.episode_cnt = 0
        self.step_cnt = 0

        self.link1_xmat = None
        self.cur_episode_len = 0

        ## reward initialization
        self.forward_reward = 0
        self.moving_reward = 0
        self.ctrl_reward = 0
        self.rotate_reward = 0
        self.velocity_reward = 0


        if self.test: ## plot reward for debugging
            self.fig1, self.ax1 = plt.subplots()
            self.xdata1, self.ydata1 = [], []
            self.ln1, = plt.plot([], [], 'r-', animated=True)

            def init1():
                self.ax1.set_xlim(0, 10000)
                self.ax1.set_ylim(-2, 2)
                self.ax1.set_xlabel("step")
                self.ax1.set_ylabel("velocity_reward")
                return self.ln1,

            def update1(frame):
                self.xdata1.append(frame)
                self.ydata1.append(self.velocity_reward)
                self.ln1.set_data(self.xdata1, self.ydata1)
                return self.ln1,

            self.ani1 = FuncAnimation(self.fig1, update1, frames=np.linspace(0, 10000, 10000), 
                                 init_func=init1, blit=True)
            
            self.fig3, self.ax3 = plt.subplots()
            self.xdata3, self.ydata3 = [], []
            self.ln3, = plt.plot([], [], 'r-', animated=True)

            def init3():
                self.ax3.set_xlim(0, 10000)
                self.ax3.set_ylim(-10, 10)
                self.ax3.set_xlabel("step")
                self.ax3.set_ylabel("pitch")
                return self.ln3,
        
            def update3(frame):
                self.xdata3.append(frame)
                self.ydata3.append(self.com_qvel[4])
                self.ln3.set_data(self.xdata3, self.ydata3)
                return self.ln3,
        
            self.ani3 = FuncAnimation(self.fig3, update3, frames=np.linspace(0, 10000, 10000), 
                                    init_func=init3, blit=True)
            
            self.fig4, self.ax4 = plt.subplots()
            self.xdata4, self.ydata4 = [], []
            self.ln4, = plt.plot([], [], 'r-', animated=True)

            def init4():
                self.ax4.set_xlim(0, 10000)
                self.ax4.set_ylim(-10, 10)
                self.ax4.set_xlabel("step")
                self.ax4.set_ylabel("yaw")
                return self.ln4,
        
            def update4(frame):
                self.xdata4.append(frame)
                self.ydata4.append(self.com_qvel[5])
                self.ln4.set_data(self.xdata4, self.ydata4)
                return self.ln4,
        
            self.ani4 = FuncAnimation(self.fig4, update4, frames=np.linspace(0, 10000, 10000), 
                                    init_func=init4, blit=True)
            
            self.fig5, self.ax5 = plt.subplots()
            self.xdata5, self.ydata5 = [], []
            self.ln5, = plt.plot([], [], 'r-', animated=True)

            def init5():
                self.ax5.set_xlim(0, 10000)
                self.ax5.set_ylim(-10, 10)
                self.ax5.set_xlabel("step")
                self.ax5.set_ylabel("roll")
                return self.ln5,
        
            def update5(frame):
                self.xdata5.append(frame)
                self.ydata5.append(self.com_qvel[3])
                self.ln5.set_data(self.xdata5, self.ydata5)
                return self.ln5,
        
            self.ani5 = FuncAnimation(self.fig5, update5, frames=np.linspace(0, 10000, 10000), 
                                    init_func=init5, blit=True)
            
            self.fig6, self.ax6 = plt.subplots()
            self.xdata6, self.ydata6 = [], []
            self.ln6, = plt.plot([], [], 'r-', animated=True)

            def init6():
                self.ax6.set_xlim(0, 10000)
                self.ax6.set_ylim(-2, 2)
                self.ax6.set_xlabel("step")
                self.ax6.set_ylabel("forward_reward")
                return self.ln6,
        
            def update6(frame):
                self.xdata6.append(frame)
                self.ydata6.append(self.forward_reward)
                self.ln6.set_data(self.xdata6, self.ydata6)
                return self.ln6,
        
            self.ani6 = FuncAnimation(self.fig6, update6, frames=np.linspace(0, 10000, 10000), interval=1,
                                    init_func=init6, blit=True)
            
            plt.show(block=False)

        if self.test or self.resume:
            self.default_step_rate = 0.5

        if self.test and self.ros:
            import rospy
            from std_msgs.msg import Float32MultiArray
            self.debug_msg = Float32MultiArray()
            self.debug_pub = rospy.Publisher('tensegrity_env/debug', Float32MultiArray, queue_size=10)

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(67,)) ## (24 + 24 + 3) * n_prev

        self.rospack = RosPack()
        
        ## change this to your own model path
        model_path = self.rospack.get_path('tensegrity_slam_sim') + '/models/scene_real_model_fullactuator.xml'
        MujocoEnv.__init__(
            self, 
            model_path, 
            2,
            observation_space=observation_space,
            **kwargs
            )
        
        utils.EzPickle.__init__(self)
    
    def step(self, action):
        self.cur_episode_len += 1
        #if self.test:
            #print("actuator force: ", action) ## (24,)
        if not self.is_params_set:
            self.set_param()
            self.is_params_set = True

        if self.prev_action is None:
            self.prev_action = [copy.deepcopy(action) for i in range(self.n_prev)]

        if self.prev_command is None:
            self.prev_command = [copy.deepcopy(self.command) for i in range(self.n_prev)]

        ## add noise to action
        self.data.qfrc_applied[:] = 0.01*self.step_rate*np.random.randn(len(self.data.qfrc_applied))

        # do simulation
        self._step_mujoco_simulation(action, self.frame_skip)

        body_xpos = np.vstack((
                    self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "link1")],
                    self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "link2")],
                    self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "link3")],
                    self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "link4")],
                    self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "link5")],
                    self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "link6")],
                    ))
        self.current_body_xpos = np.mean(body_xpos, axis=0) ## (3,)
        body_xquat = np.concatenate([
                    self.data.xquat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link1")],
                    self.data.xquat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link2")],
                    self.data.xquat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link3")],
                    self.data.xquat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link4")],
                    self.data.xquat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link5")],
                    self.data.xquat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link6")],
                    ])
        self.current_body_xquat = copy.deepcopy(body_xquat) ## (24,)
        self.body_qvel = np.vstack([
                    self.data.qvel[0:6],
                    self.data.qvel[6:12],
                    self.data.qvel[12:18],
                    self.data.qvel[18:24],
                    self.data.qvel[24:30],
                    self.data.qvel[30:36],
                    ])
        self.com_qvel = np.mean(self.body_qvel, axis=0)
        self.angular_velocity = self.body_qvel[:, 3:6]
        
        # initialize reward
        self.forward_reward = 0
        self.moving_reward = 0
        self.ctrl_reward = 0
        self.rotate_reward = 0
        self.velocity_reward = 0
        
        ## calculate forward reward
        #if self.com_qvel[3] > 1.0: # pitch > 2.0
        #    self.rotate_reward = 0.1*1.0
        #elif self.com_qvel[3] < 0.0: # pitch < 0.0 
        #    self.rotate_reward = -0.1*np.abs(1.0*(1.0-self.com_qvel[3])) # -||des_pitch - real_pitch||
        #else:
        #    self.rotate_reward = 0.1*np.exp(-1.0*np.square(1.0-self.com_qvel[3])) # exp(-20*||des_pitch - real_pitch||^2)
        self.forward_reward = 100.0*(self.current_body_xpos[0] - self.prev_body_xpos[0])[0]
        
        ## calculate velocity reward
        desired_velocity = self.command[1]
        if self.com_qvel[4] > 0:
            self.velocity_reward = np.exp(-4.0*np.abs(desired_velocity-self.com_qvel[4]))
        else:
            self.velocity_reward = -0.2*np.abs(self.com_qvel[4]) # -0.2*||pitch||
        """
        if np.dot(desired_velocity, self.com_qvel[0:2]) > np.square(desired_velocity).sum(): # if des_vel * com_vel > abs(des_vel)^2
            self.velocity_reward = 1.0
        elif np.dot(desired_velocity, self.com_qvel[0:2]) < 0.0: # if des_vel * com_vel < 0
            self.velocity_reward = -1.0*np.abs(1.0*(desired_velocity[0]-self.com_qvel[0])) # -||des_vel - com_vel||
        else:
            self.velocity_reward = np.exp(-20.0*np.square(desired_velocity-self.com_qvel[0:2]).sum()) # exp(-20*||des_vel - com_vel||^2)
        """

        
        ## calculate forward reward
        #if self.command is not None:
        #    forward_reward = 10.0*np.dot(self.current_body_xpos[:2] - self.prev_body_xpos[-1][:2], np.array([np.cos(np.deg2rad(self.command)), np.sin(np.deg2rad(self.command))]))
        #else:
        #    raise ValueError("command is not set")
        
        
        if self.test:
            ## draw reward for debugging
            print("cur_ep_cnt: ", self.cur_episode_len)
            self.fig3.canvas.draw()
            self.fig3.canvas.flush_events()
        
        ## calculate moving reward
        #moving_reward = 10.0*np.linalg.norm(self.current_body_xpos - self.prev_body_xpos[-1])
        #ctrl_reward = -0.1*self.step_rate*np.linalg.norm(action-self.prev_action[-1])
        #print("rotate_reward: ", self.rotate_reward)
        #print("velocity_reward: ", self.velocity_reward)
        reward = 0.65*self.forward_reward + 0.35*self.velocity_reward
       
        self.episode_cnt += 1
        self.step_cnt += 1

        truncated = False
        terminated = not (self.episode_cnt < self.max_episode)

        self.prev_body_xpos.append(copy.deepcopy(self.current_body_xpos)) ## (3,)
        if len(self.prev_body_xpos) > self.n_prev:
            self.prev_body_xpos.pop(0)
        self.prev_body_xquat.append(copy.deepcopy(self.current_body_xquat)) ## (24,)
        if len(self.prev_body_xquat) > self.n_prev:
            self.prev_body_xquat.pop(0)
        if len(self.prev_body_xpos) > self.n_prev:
            self.prev_body_xpos.pop(0)
        self.prev_action.append(copy.deepcopy(action)) ## (24,)
        if len(self.prev_action) > self.n_prev:
            self.prev_action.pop(0)

        ## convert world command to local command
        #if self.command is not None:
        #    self.local_command = self.world_to_local(self.command)
        #else:
        #    raise ValueError("command is not set")


        ## observation
        assert self.command is not None
        assert self.prev_command is not None
        obs = self._get_obs()
        
        if terminated or truncated:
            self.episode_cnt = 0
            self.current_body_xpos = None
            self.prev_body_xpos = None
            self.current_body_xquat = None
            self.prev_body_xquat = None
            self.prev_action = None

        return (
            obs,
            reward,
            terminated,
            truncated,
            dict(
                reward_forward=self.forward_reward,
                reward_moving=self.moving_reward,
                reward_ctrl=self.ctrl_reward,
                reward_rotate=self.rotate_reward,
                reward_velocity=self.velocity_reward,
                )
        )
    
    def world_to_local(self, world_command):
        ## convert world command to local command
        self.link1_xmat = self.data.xmat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link1")].reshape(3, 3)
        self.local_command = np.dot(self.link1_xmat.T, np.array(world_command).T)
        return self.local_command
    
    def _get_obs(self):
        return np.concatenate( ## (67,)
            [
                np.concatenate(self.prev_body_xquat), ## (24,)
                np.concatenate(self.angular_velocity), ## (18,)
                np.concatenate(self.prev_action), ## (24,)
                self.command[1], ## (1,)
            ]
        )
    
    def reset_model(self):
        self.cur_episode_len = 0
        if self.test or self.resume:
            self.step_rate = self.default_step_rate
        elif self.max_step:
            self.step_rate = min(float(self.step_cnt)/self.step_rate_max_cnt, 1)
        self.max_episode = 500 + 1500*self.step_rate

        qpos = np.array([-0.1, 0, 0.0, 1.0, 0, 0, 0,
                0.1, 0, 0.0, 1.0, 0, 0, 0,
                0, 0.1, 0.0, 1.0, 0, 0, 0,
                0, -0.1, 0.0, 1.0, 0, 0, 0,
                0, 0, 0.1, 1.0, 0, 0, 0,
                0, 0, -0.1, 1.0, 0, 0, 0
                ])
        qpos += 0.02*self.step_rate*np.random.randn(len(qpos))
        ## add initial velocity
        qvel = self.init_qvel
        if self.randomize_position and self.test:
            qpos += np.array([0, 0, 0.5, 0, 0, 0, 0,
                0, 0, 0.5, 0, 0, 0, 0,
                0, 0, 0.5, 0, 0, 0, 0,
                0, 0, 0.5, 0, 0, 0, 0,
                0, 0, 0.5, 0, 0, 0, 0,
                0, 0, 0.5, 0, 0, 0, 0
                ])
            qvel += 0.02*self.step_rate*np.random.randn(len(qvel))
        self.set_state(qpos, qvel)

        if (self.prev_body_xquat is None) and (self.prev_action is None):
            self.current_qvel = self.data.qvel.flat[:]
            body_xpos = np.vstack((
                    self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "link1")],
                    self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "link2")],
                    self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "link3")],
                    self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "link4")],
                    self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "link5")],
                    self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "link6")],
                    ))
            self.prev_body_xpos = [np.mean(body_xpos, axis=0) for i in range(self.n_prev)]
            body_xquat = np.concatenate([
                        self.data.xquat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link1")],
                        self.data.xquat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link2")],
                        self.data.xquat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link3")],
                        self.data.xquat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link4")],
                        self.data.xquat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link5")],
                        self.data.xquat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link6")],
                        ])
            self.prev_body_xquat = [copy.deepcopy(body_xquat) for i in range(self.n_prev)]
            self.prev_action = [np.zeros(self.action_length) for i in range(self.n_prev)] ## (12,)
            self.angular_velocity = self.data.qvel.reshape(6, 6)[:, 3:6]
        
        ## switch to new command
        if self.test:
            self.command = [0.0, 1.0, 0.0]
            #self.command = np.random.uniform(-180, 180)
        else:
            v = np.random.uniform(0.5, 0.75 + 1*self.step_rate)            
            self.command = [0.0, v, 0.0]
        
        self.prev_command = [self.command for i in range(self.n_prev)] ## (1,)

        return self._get_obs()