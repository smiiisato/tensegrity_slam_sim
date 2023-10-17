## if the reward is beyond the threshold, then start randomizing the initial position

from stable_baselines3.common.callbacks import BaseCallback

class StartRandomizingCallback(BaseCallback):
    """
    平均報酬が閾値を超えたら、初期位置のランダム化を開始するコールバック。
    """
    def __init__(self, threshold, verbose=0):
        super(StartRandomizingCallback, self).__init__(verbose)
        self.threshold = threshold
        self.changed = False
        self.episode_rewards = None

    def _on_step(self) -> bool:
        """
        This method is called before collecting the rollouts.
        """
        self.ep_rew_mean = self.logger.info("rollout/ep_rew_mean")
        if self.ep_rew_mean and (self.ep_rew_mean > self.threshold) and not self.changed:
            self.training_env.env_method("start_randomizing_position")
            self.changed = True
        
        return True

        
        
