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

class TensegrityEnvRealmodelFullactuatorAngularmomentum(TensegrityEnv):

    def __init__(self, act_range=6.0, test=False, ros=False, max_steps=None, resume=False, **kwargs):
        self.action_length = 24
        self.is_params_set = False
        self.test = test
        self.ros = ros
        self.max_step = max_steps
        self.step_rate_max_cnt = 50000000
        self.resume = resume
        self.act_range = act_range
        print("act_range: ", self.act_range)

        # control range
        self.ctrl_max = [0]*self.action_length
        self.ctrl_min = [-self.act_range]*self.action_length

        # initial command, direction +x
        self.command = [0.5, 0.0, 0.0]
        self.local_command = None

        # flag for randomizing initial position
        #self.randomize_position = (self.resume or self.test)
        self.randomize_position = False

        self.n_prev = 1
        self.max_episode = 1000
    
        self.current_body_xpos = None
        self.current_body_xquat = None
        self.prev_body_xpos = None
        self.prev_body_xquat = None
        self.prev_action = None

        self.episode_cnt = 0
        self.step_cnt = 0

        self.link1_xmat = None

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
                self.ax1.set_xlim(0, 100)
                self.ax1.set_ylim(-2, 2)
                self.ax1.set_xlabel("step")
                self.ax1.set_ylabel("velocity_reward")
                return self.ln1,

            def update1(frame):
                self.xdata1.append(frame)
                self.ydata1.append(self.velocity_reward)
                self.ln1.set_data(self.xdata1, self.ydata1)
                return self.ln1,

            self.ani1 = FuncAnimation(self.fig1, update1, frames=np.linspace(0, 100, 1000),
                                 init_func=init1, blit=True)
            
            self.fig2, self.ax2 = plt.subplots()
            self.xdata2, self.ydata2 = [], []
            self.ln2, = plt.plot([], [], 'r-', animated=True)

            def init2():
                self.ax2.set_xlim(0, 100)
                self.ax2.set_ylim(-2, 2)
                self.ax2.set_xlabel("step")
                self.ax2.set_ylabel("rotate_reward")
                return self.ln2,
    
            def update2(frame):
                self.xdata2.append(frame)
                self.ydata2.append(self.rotate_reward)
                self.ln2.set_data(self.xdata2, self.ydata2)
                return self.ln2,
                
            self.ani2 = FuncAnimation(self.fig2, update2, frames=np.linspace(0, 100, 1000),
                                    init_func=init2, blit=True)
            
            plt.show(block=False)

        if self.test or self.resume:
            self.default_step_rate = 0.5

        if self.test and self.ros:
            import rospy
            from std_msgs.msg import Float32MultiArray
            self.debug_msg = Float32MultiArray()
            self.debug_pub = rospy.Publisher('tensegrity_env/debug', Float32MultiArray, queue_size=10)

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(27,)) ## (24 + 24 + 3) * n_prev

        self.rospack = RosPack()
        
        ## change this to your own model path
        model_path = self.rospack.get_path('tensegrity_slam_sim') + '/models/scene_real_model_fullactuator.xml'
        MujocoEnv.__init__(
            self, 
            model_path, 
            5,
            observation_space=observation_space,
            **kwargs
            )
        
        utils.EzPickle.__init__(self)
    
    def step(self, action):
        if self.test:
            print("actuator force: ", action) ## (24,)
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
        
        # initialize reward
        self.forward_reward = 0
        self.moving_reward = 0
        self.ctrl_reward = 0
        self.rotate_reward = 0
        self.velocity_reward = 0

        ## calculate angular momentum
        angular_momentum = np.zeros(3)
        self.com_qvel = np.mean(self.body_qvel, axis=0)[0:3]
        self.com_xpos = self.current_body_xpos

        for i in range(6):
            body_mass = self.model.body_mass[i]
            body_com_xpos = body_xpos[i]
            body_vel = self.body_qvel[i][0:3]
            angular_momentum += body_mass*np.cross(body_com_xpos-self.com_xpos, body_vel-self.com_qvel) # m_i * ((r_i-r) x (v_i-v))

        ## calculate rotate reward
        self.rotate_reward = 10.0*angular_momentum[2]

        ## calculate velocity reward
        desired_velocity = self.command[0:2]
        if np.dot(desired_velocity, self.com_qvel[0:2]) > np.square(desired_velocity).sum(): # if des_vel * com_vel > abs(des_vel)^2
            self.velocity_reward = 1.0
        else:
            self.velocity_reward = np.exp(-20.0*np.square(desired_velocity-self.com_qvel[0:2]).sum()) # exp(-20*||des_vel - com_vel||^2)

        
        ## calculate forward reward
        #if self.command is not None:
        #    forward_reward = 10.0*np.dot(self.current_body_xpos[:2] - self.prev_body_xpos[-1][:2], np.array([np.cos(np.deg2rad(self.command)), np.sin(np.deg2rad(self.command))]))
        #else:
        #    raise ValueError("command is not set")
        
        if self.test:
            print("rotate_reward: ", self.rotate_reward)
            print("velocity_reward: ", self.velocity_reward)
            ## draw reward for debugging
            self.fig1.canvas.draw()
            self.fig1.canvas.flush_events()
            self.fig2.canvas.draw()
            self.fig2.canvas.flush_events()
        
        ## calculate moving reward
        #moving_reward = 10.0*np.linalg.norm(self.current_body_xpos - self.prev_body_xpos[-1])
        #ctrl_reward = -0.1*self.step_rate*np.linalg.norm(action-self.prev_action[-1])
        print("rotate_reward: ", self.rotate_reward)
        print("velocity_reward: ", self.velocity_reward)
        reward = self.forward_reward + self.moving_reward + self.ctrl_reward + self.rotate_reward + self.velocity_reward

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
        if self.command is not None:
            self.local_command = self.world_to_local(self.command)
        else:
            raise ValueError("command is not set")


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
        return np.concatenate(
            [
                #np.concatenate(self.prev_body_xquat),
                np.concatenate(self.prev_action),
                self.local_command,
            ]
        )
    
    def reset_model(self):
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
        
        ## switch to new command
        if self.test:
            self.command = [0.5, 0.0, 0.0]
            #self.command = np.random.uniform(-180, 180)
        else:
            self.command = [0.5, 0.0, 0.0]
        
        self.prev_command = [self.command for i in range(self.n_prev)] ## (1,)
        self.local_command = self.world_to_local(self.command)

        return self._get_obs()