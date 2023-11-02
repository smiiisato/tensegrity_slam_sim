## if the reward is beyond the threshold, then finish this learning

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

class StartRandomizingCallback(BaseCallback):
    """
    平均報酬が閾値を超えたら、初期位置のランダム化を開始するコールバック。
    """
    def __init__(self, threshold=1000, env, model, verbose=0):
        super(RewardThresholdCallback, self).__init__(verbose)
        self.threshold = threshold
        self.ep_rew_mean = None
        self.env = env
        self.model = model

    def _on_step(self) -> bool:
        """
        This method is called before collecting the rollouts.
        """
        self.ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        if self.ep_rew_mean and (self.ep_rew_mean > self.threshold):
            print("stopped training at reward mean of {}".format(self.ep_rew_mean))
            return False
        
        return True
