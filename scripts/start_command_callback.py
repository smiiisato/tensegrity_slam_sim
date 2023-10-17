## if the reward is beyond the threshold, then start randomizing the command

from stable_baselines3.common.callbacks import BaseCallback

class StartCommandCallback(BaseCallback):
    def __init__(self, threshold, verbose=0):
        super(StartCommandCallback, self).__init__(verbose)
        self.threshold = threshold
        self.changed = False
        self.ep_rew_mean = None

    def _on_step(self) -> bool:
        """
        This method is called before collecting the rollouts.
        """
        self.ep_rew_mean = self.logger.info("rollout/ep_rew_mean")
        if self.ep_rew_mean and (self.ep_rew_mean > self.threshold) and not self.changed:
            self.training_env.env_method("start_randomizing_command")
            self.changed = True
        
        return True