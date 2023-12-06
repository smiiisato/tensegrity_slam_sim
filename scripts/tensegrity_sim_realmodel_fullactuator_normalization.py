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
from tensegrity_sim_realmodel_fullactuator_angularvelocity import TensegrityEnvRealmodelFullactuatorAngularvelocity

class TensegrityEnvRealmodelFullactuatorNormalization(TensegrityEnvRealmodelFullactuatorAngularvelocity):

    def step(self, action):
        # rescale action from [-1, 1] to [low, high]
        action = self.rescale_actions(action, np.array(self.ctrl_min), np.array(self.ctrl_max))
        return super().step(action)
    
    def _get_obs(self):
        return np.concatenate(
            [
                np.concatenate(self.prev_body_xquat), ## (24,)
                np.concatenate(self.angular_velocity), ## (18,)
                np.concatenate(self.normalize_action(self.prev_action)), ## (24,)
                np.array([self.command[1]]), ## (1,)
            ]
        )
    
    def normalize_action(self, action):
        # normalize action from [low, high] to [-1, 1]
        normalized_action = (np.array(action) - np.array(self.ctrl_min)) / (np.array(self.ctrl_max) - np.array(self.ctrl_min)) * 2 - 1
        return normalized_action
    
    def normalize_xquat(self, xquat):
        # normalize xquat 
        norm = np.linalg.norm(xquat)
        if norm == 0: 
            return np.array([1, 0, 0, 0])
        return xquat / norm
    
    def rescale_actions(self, action, low, high):
        # rescale action from [-1, 1] to [low, high]
        rescaled_action = (action + 1) / 2 * (high - low) + low
        return rescaled_action
    
    def _set_action_space(self):
        low = -1.0*np.ones(24)
        high = 1.0*np.ones(24)
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        return self.action_space

