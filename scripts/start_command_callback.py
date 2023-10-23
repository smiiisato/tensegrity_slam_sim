## if the reward is beyond the threshold, then start randomizing the command

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

class StartCommandCallback(BaseCallback):
    """
    平均報酬が閾値を超えたら、初期位置のランダム化を開始するコールバック。
    """
    def __init__(self, threshold, model, verbose=0):
        super(StartCommandCallback, self).__init__(verbose)
        self.threshold = threshold
        self.randomized = False
        self.ep_rew_mean = None
        self.model = model
        self.current_command = 0
     
    def _on_step(self) -> bool:
        """
        This method is called before collecting the rollouts.
        """
        self.ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        if self.ep_rew_mean and (self.ep_rew_mean > self.threshold) and not self.randomized:
            self.current_command += 1
            self.model.get_env().set_attr("command", self.current_command)

        if self.current_command > 2 and (self.ep_rew_mean > self.threshold) and not self.randomized:
            self.randomized = True
            self.model.get_env().set_attr("randomize_command", True)
        
        return True