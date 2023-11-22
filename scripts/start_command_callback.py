## if the reward is beyond the threshold, then start randomizing the command

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
import numpy as np

class StartCommandCallback(BaseCallback):
    """
    平均報酬が閾値を超えたら、初期位置のランダム化を開始するコールバック。
    """
    def __init__(self, threshold, env, model, verbose=0):
        super(StartCommandCallback, self).__init__(verbose)
        self.threshold = threshold
        self.randomized = False
        self.ep_rew_mean = None
        self.env = env
        self.current_command = 0
        self.model = model
        self.steps_since_increment = 0
        self.increment_amount = 100000
     
    def _on_step(self) -> bool:
        """
        This method is called before collecting the rollouts.
        """
        self.ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        if self.ep_rew_mean and (self.ep_rew_mean > self.threshold):
            if not self.randomized:
                self.randomized = True
            elif self.steps_since_increment > self.increment_amount:
                self.steps_since_increment = 0
                self.training_env.env_method("enlarge_command_space")
        if self.randomized:
            self.steps_since_increment += 1
        
        return True