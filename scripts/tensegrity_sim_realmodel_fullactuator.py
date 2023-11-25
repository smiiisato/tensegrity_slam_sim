import copy
from typing import Any, Optional, SupportsFloat
import mujoco
import numpy as np
from rospkg import RosPack
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv
from tensegrity_sim import TensegrityEnv

class TensegrityEnvRealmodelFullactuator(TensegrityEnv):

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
        return super().step(action)
    
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
            self.command = 0
            #self.command = np.random.uniform(-180, 180)
        else:
            self.command = 0
        
        self.prev_command = [self.command for i in range(self.n_prev)] ## (1,)

        return self._get_obs()