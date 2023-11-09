import copy
from typing import Any, Optional, SupportsFloat
import mujoco
import numpy as np
from rospkg import RosPack
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv
from tensegrity_sim import TensegrityEnv

class TensegrityEnvDirection(TensegrityEnv):
    
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
        if self.randomize_position:
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
            self.prev_action = [np.zeros(12) for i in range(self.n_prev)] ## (12,)
        
        ## switch to new command
        if self.test:
            self.command = 0
            #self.command = np.random.uniform(-180, 180)
        else:
            self.command = 0
        
        self.prev_command = [self.command for i in range(self.n_prev)] ## (1,)

        return self._get_obs()