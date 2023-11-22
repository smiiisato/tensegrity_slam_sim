import copy
from typing import Any, Optional, SupportsFloat
import mujoco
import numpy as np
from rospkg import RosPack
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv
from tensegrity_sim import TensegrityEnv

class TensegrityEnvVelocity(TensegrityEnv):

    def __init__(self, act_range=6.0, test=False, ros=False, max_steps=None, resume=False, **kwargs):
        self.is_params_set = False
        self.test = test
        self.ros = ros
        self.max_step = max_steps
        self.resume = resume
        self.act_range = act_range
        
        # flag for randomizing initial position
        self.randomize_position = False

        # flag for randomizing command
        self.randomize_command = False

        # control range
        self.ctrl_max = [0]*24
        self.ctrl_min = [-act_range]*24

        # initial command, direction +x
        self.command = np.array([0.5, 0])
        # max degree of command range
        self.max_degree = 0

        self.n_prev = 3
        self.max_episode = 1000
        self.max_command = 500

        self.step_rate_max_cnt = 50000000
        
        self.current_body_xpos = None
        self.current_body_xquat = None
        self.current_body_xyvel = None
        self.prev_body_xpos = None
        self.prev_body_xquat = None
        self.prev_body_xyvel = None
        self.prev_action = None
        self.prev_command = None

        self.episode_cnt = 0
        self.step_cnt = 0
        self.command_cnt = 0

        if self.test or self.resume:
            self.default_step_rate = 0.5

        if self.test and self.ros:
            import rospy
            from std_msgs.msg import Float32MultiArray
            self.debug_msg = Float32MultiArray()
            self.debug_pub = rospy.Publisher('tensegrity_env/debug', Float32MultiArray, queue_size=10)

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(159,)) ## (24 + 24 + 3) * n_prev + prev_command = 

        self.rospack = RosPack()
        model_path = self.rospack.get_path('tensegrity_slam_sim') + '/models/scene.xml'
        MujocoEnv.__init__(
            self, 
            model_path, 
            5,
            observation_space=observation_space,
            **kwargs
            )
        
        utils.EzPickle.__init__(self)

    def step(self, action): ## action: (24,) tention of each cable
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
                    self.data.xpos[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link1")]],
                    self.data.xpos[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link2")]],
                    self.data.xpos[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link3")]],
                    self.data.xpos[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link4")]],
                    self.data.xpos[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link5")]],
                    self.data.xpos[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link6")]],
                    ))
        self.current_body_xpos = np.mean(body_xpos, axis=0) ## (3,)
        body_xquat = np.concatenate([
                    self.data.xquat[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link1")]],
                    self.data.xquat[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link2")]],
                    self.data.xquat[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link3")]],
                    self.data.xquat[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link4")]],
                    self.data.xquat[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link5")]],
                    self.data.xquat[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link6")]],
                    ])
        self.current_body_xquat = copy.deepcopy(body_xquat) ## (24,)
        body_xyvel = np.vstack((
                    self.data.qvel[0:2],
                    self.data.qvel[6:8],
                    self.data.qvel[12:14],
                    self.data.qvel[18:20],
                    self.data.qvel[24:26],
                    self.data.qvel[30:32],
                    ))
        self.current_body_xyvel = np.mean(body_xyvel, axis=0) ## (6,)
        #print(self.current_body_xyvel)

        forward_reward = 0
        moving_reward = 0
        ctrl_reward = 0
        # reward
        if self.command is not None:
            if np.dot(self.current_body_xyvel, self.command) > np.linalg.norm(self.command):
                forward_reward = 1.0
            else:
                forward_reward = np.exp(-20*(np.dot(self.current_body_xyvel, self.command) - np.linalg.norm(self.command))**2)
        else:
            raise ValueError("command is not set")
        #moving_reward = 10.0*np.linalg.norm(self.current_body_xpos - self.prev_body_xpos[-1])
        ##ctrl_reward = -0.1*self.step_rate*np.linalg.norm(action-self.prev_action[-1])
        reward = forward_reward + moving_reward + ctrl_reward
        print("reward: ", reward)

        if self.test:
            print("forward_reward: ", forward_reward)
            print("current_body_xyvel: ", self.current_body_xyvel)

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
                reward_forward=forward_reward,
                reward_moving=moving_reward,
                reward_ctrl=ctrl_reward,
                )
            )

    def _get_obs(self):
        return np.concatenate(
            [
                np.concatenate(self.prev_body_xquat),
                np.concatenate(self.prev_action),
                np.concatenate(self.prev_body_xpos),
                np.concatenate(self.prev_command),
            ]
        )
    
    def enlarge_command_space(self):
        ## enlarge command range: max[-180, 180]
        self.randomize_command = True
        if self.max_degree < 180:
            self.max_degree += 20
        return
    
    def reset_model(self):
        if self.max_step:
            self.step_rate = min(float(self.step_cnt)/self.step_rate_max_cnt, 1)
        elif self.test:
            self.step_rate = self.default_step_rate
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
        if self.randomize_position or self.test:
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
                    self.data.xpos[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link1")]],
                    self.data.xpos[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link2")]],
                    self.data.xpos[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link3")]],
                    self.data.xpos[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link4")]],
                    self.data.xpos[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link5")]],
                    self.data.xpos[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link6")]],
                    ))
            self.prev_body_xpos = [np.mean(body_xpos, axis=0) for i in range(self.n_prev)]
            body_xquat = np.concatenate([
                        self.data.xquat[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link1")]],
                        self.data.xquat[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link2")]],
                        self.data.xquat[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link3")]],
                        self.data.xquat[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link4")]],
                        self.data.xquat[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link5")]],
                        self.data.xquat[self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link6")]],
                        ])
            self.prev_body_xquat = [copy.deepcopy(body_xquat) for i in range(self.n_prev)]
            self.prev_action = [np.zeros(24) for i in range(self.n_prev)] ## (24,)
        
        ## switch to new command
        if self.test:
            #self.command = 0
            self.command = np.array([0.5, 0])
        else:
            if self.randomize_command:
                #self.command = np.array([np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8)])
                self.command = np.array([0.5, 0])
            else:
                self.command = np.array([0.5, 0])
        
        self.prev_command = [self.command for i in range(self.n_prev)] ## (1,)
        print("command: ", self.command)

        return self._get_obs()