import copy
from typing import Any, Optional, SupportsFloat
import mujoco
import numpy as np
from rospkg import RosPack
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv

class TensegrityEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, act_range=6.0, test=False, ros=False, max_steps=None, resume=False, **kwargs):
        self.is_params_set = False
        self.test = test
        self.ros = ros
        self.max_step = max_steps
        self.step_rate_max_cnt = 50000000
        self.resume = resume
        self.act_range = act_range

        # control range
        self.ctrl_max = [0]*24
        self.ctrl_min = [-self.act_range]*24

        # initial command, direction +x
        self.command = 0

        # flag for randomizing initial position
        self.randomize_position = (self.resume or self.test)

        self.n_prev = 3
        self.max_episode = 1000
        
        self.current_body_xpos = None
        self.current_body_xquat = None
        self.prev_body_xpos = None
        self.prev_body_xquat = None
        self.prev_action = None

        self.episode_cnt = 0
        self.step_cnt = 0

        if self.test or self.resume:
            self.default_step_rate = 0.5

        if self.test and self.ros:
            import rospy
            from std_msgs.msg import Float32MultiArray
            self.debug_msg = Float32MultiArray()
            self.debug_pub = rospy.Publisher('tensegrity_env/debug', Float32MultiArray, queue_size=10)

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(153,)) ## (24 + 24 + 3) * n_prev

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

    def set_param(self):
        if self.test:
            self.mujoco_renderer.viewer._render_every_frame = False

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

        forward_reward = 0
        moving_reward = 0
        ctrl_reward = 0
        # reward
        if self.command is not None:
            forward_reward = 100.0*np.dot(self.current_body_xpos[:2] - self.prev_body_xpos[-1][:2], np.array([np.cos(np.deg2rad(self.command)), np.sin(np.deg2rad(self.command))]))
        else:
            raise ValueError("command is not set")
        moving_reward = 10.0*np.linalg.norm(self.current_body_xpos - self.prev_body_xpos[-1])
        ##ctrl_reward = -0.1*self.step_rate*np.linalg.norm(action-self.prev_action[-1])
        reward = forward_reward + moving_reward + ctrl_reward

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
                np.concatenate(self.prev_body_xpos)
            ]
        )
    
    def _set_action_space(self):
        low = np.asarray(self.ctrl_min)
        high = np.asarray(self.ctrl_max)
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        return self.action_space
    
    def start_randomizing_position(self):
        self.randomize_position = True
        return

    def enlarge_command_space(self):
        return
    
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

            if self.command is None:
                self.command = 0

        return self._get_obs()

        