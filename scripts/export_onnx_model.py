import torch as th
import numpy as np
import pickle
import os
from stable_baselines3 import PPO


class RunningMeanStd(th.nn.Module):
    def __init__(self, mean, var, obs_clip_range):

        super().__init__()
        self.mean = th.tensor(mean, dtype=th.float32)
        self.var = th.tensor(var, dtype=th.float32)
        self.obs_clip_range = obs_clip_range

    def forward(self, input):
        obs_normalized = th.clamp((th.tensor(input, dtype=th.float32) - self.mean) / th.sqrt(self.var + 1e-8),
                                  -self.obs_clip_range, self.obs_clip_range).float()
        return obs_normalized


class OnnxablePolicyWithObsNormalize(th.nn.Module):
    def __init__(self, extractor, action_net, value_net, obs_rms_mean, obs_rms_var, obs_clip_range):
        super().__init__()
        self.extractor = extractor
        self.action_net = action_net
        self.value_net = value_net
        self.obs_rms = RunningMeanStd(obs_rms_mean, obs_rms_var, obs_clip_range)

    def forward(self, observation):
        obs_normalized = self.obs_rms(observation)
        action_hidden, value_hidden = self.extractor(obs_normalized)
        return self.action_net(action_hidden)


trial = 279
load_step = 242073600
obs_clip_range = 100

root_dir = os.path.dirname(os.path.abspath(__file__))
model_path = root_dir + "/../saved/PPO_{0}/models/model_{1}_steps.zip".format(trial, load_step)
stats_file = f'{root_dir}/../saved/PPO_{trial}/models/model_vecnormalize_{load_step}_steps.pkl'
model = PPO.load(model_path, device="cpu")
with open(stats_file, 'rb') as f:
    stats_data = pickle.load(f)
obs_rms_mean = stats_data.obs_rms.mean
obs_rms_var = stats_data.obs_rms.var
onnxable_model_with_obs_normalization = OnnxablePolicyWithObsNormalize(model.policy.mlp_extractor,
                                                                        model.policy.action_net,
                                                                        model.policy.value_net,
                                                                        obs_rms_mean,
                                                                        obs_rms_var,
                                                                       obs_clip_range=obs_clip_range)

observation_size = model.observation_space.shape
dummy_input = th.randn(1, *observation_size, dtype=th.float32)

th.onnx.export(
    onnxable_model_with_obs_normalization,
    dummy_input,
    root_dir + "/../saved/PPO_{0}/models/model_{1}_steps_with_obs_normalize.onnx".format(trial, load_step),
    input_names=["input"],
    opset_version=9,)

# Load example and test with onnxruntime inference
import onnx
import onnxruntime as ort


onnx_path = root_dir + "/../saved/PPO_{0}/models/model_{1}_steps_with_obs_normalize.onnx".format(trial, load_step)
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

observation = -2.345*np.ones((1, *observation_size)).astype(np.float32)

ort_sess = ort.InferenceSession(onnx_path)
# input_names = [input.name for input in ort_sess.get_inputs()]
# print("Input Names:", input_names)
action = ort_sess.run(None, {"input": observation})
print("action", action)
print("export onnx file and check validation finished!")



