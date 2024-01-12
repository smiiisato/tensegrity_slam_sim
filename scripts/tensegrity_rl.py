import os
import re
import time
import glob
import argparse
import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed, get_device, get_latest_run_id
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tensegrity_sim import TensegrityEnv
from tensegrity_sim_realmodel_fullactuator_velocity import TensegrityEnvRealModelFullActuatorVelocity
from tensegrity_sim_realmodel_fullactuator_angularmoment import TensegrityEnvRealModelFullActuatorAngularMomentum
from tensegrity_sim_realmodel_fullactuator_linear_velocity import TensegrityEnvRealModelFullActuatorLinearVelocity
from tensegrity_sim_realmodel_fullactuator_no_stiffness import TensegrityEnvRealModelFullActuatorNoStiffness
from tensegrity_sim_realmodel_fullactuator_no_stiffness_imu import TensegrityEnvRealModelFullActuatorNoStiffnessImu
from tensegrity_sim_realmodel_fullactuator_no_stiffness_init_pos import TensegrityEnvRealModelFullActuatorNoStiffnessInitPos


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2, help="Random seed")
    parser.add_argument("--what", type=str, default="train", help="what do you want to do: [train, test]")
    parser.add_argument("--sim_env", type=int, default=1,
                        help="type of simulation environment: [1(normal), 2(realmodel_full_vel)]")

    # observation and action space
    parser.add_argument("--normalize_obs", action="store_true", default=True, help="whether normalize the obs value or not")
    parser.add_argument("--obs_range", type=float, default=100.0, help="actuator control range")  # max obs value before sending to moving average
    parser.add_argument("--act_range", type=float, default=16.0, help="actuator control range")  # action space(force value) TODO:normalize action

    # learning-related params
    parser.add_argument("--n_env", type=int, default=1, help="number of sub_env/parallel_env to use")
    parser.add_argument("--batch_size", type=int, default=24576, help="number of batch size(experience buffer size)")  # experience buffer size
    parser.add_argument("--minibatch", type=int, default=2048, help="number of mini_batch to update policy")  # minibatch size
    parser.add_argument("--epoch", type=int, default=5, help="number of epoch to update")  # data epoch numbers for one policy iteration
    parser.add_argument("--max_step", type=int, default=400000000, help="PPO train total time steps")  # sum of all parallel envs' steps
    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--save_interval", type=int, default=50, help="interval of iteration to save network weight")
    parser.add_argument("--net_layer", type=int, default=5, help="number of layer of network") # network layer number

    # test-related params
    parser.add_argument("--render", type=int, default=1, help="Render or not (only when testing)")
    parser.add_argument("--trial", type=int, default=None, help="PPO trial number to find the policy directory")
    parser.add_argument("--load_step", type=int, default=None, help="step identifier to load specific saved policy model")
    parser.add_argument("--wait", action="store_true", help="wait of render when testing")
    parser.add_argument('--resume', action="store_true", default=False, help='resume the training')
    parser.add_argument('--ros', action="store_true", help='publish some info using ros when testing')
    parser.add_argument("--best_rate", type=float, default=0.0, help="if 0.0, choose best snapshot from all iterations")
    return parser


def make_env(test, max_step, act_range=6.0, resume=False, render_mode=None):
    args = parser().parse_args()

    def _init():
        env_class_options = [TensegrityEnv, 
                             TensegrityEnvRealModelFullActuatorVelocity, 
                             TensegrityEnvRealModelFullActuatorAngularMomentum,
                             TensegrityEnvRealModelFullActuatorLinearVelocity,
                             TensegrityEnvRealModelFullActuatorNoStiffness,
                             TensegrityEnvRealModelFullActuatorNoStiffnessImu,
                             TensegrityEnvRealModelFullActuatorNoStiffnessInitPos]  # TODO: add new env class here
        env_cls = env_class_options[args.sim_env-1]
        info_key = env_cls.info_keywords
        # print(info_key)
        # create basic env with monitor( haven't been vectorized)
        # env = Monitor(env_cls(test=test, max_steps=max_step, resume=resume, act_range=act_range, render_mode=render_mode), filename="../saved/reward_csv/result", info_keywords=info_key)
        env = Monitor(env_cls(test=test, max_steps=max_step, resume=resume, act_range=act_range, render_mode=render_mode))
        assert env is not None, "env is None"
        return env
    return _init


def main():
    args = parser().parse_args()
    set_random_seed(args.seed)
    root_dir = os.path.dirname(os.path.abspath(__file__))

    max_step_per_sub_env = int(args.max_step - 1 / args.n_env) + 1
    if args.what == "train":
        env = SubprocVecEnv([make_env(test=False,
                                      max_step=max_step_per_sub_env,
                                      act_range=args.act_range,
                                      resume=args.resume,
                                      render_mode=None) for _ in range(args.n_env)])
        # normalize obs if is needed
        assert isinstance(env, VecEnv)
        if args.normalize_obs:
            env = VecNormalize(venv=env,
                               training=True,
                               norm_obs=True,
                               norm_reward=False,
                               clip_obs=args.obs_range,
                               gamma=args.gamma)
    else:  # test mode
        assert args.n_env == 1
        env = DummyVecEnv([make_env(test=True,
                                    max_step=None,
                                    act_range=args.act_range,
                                    resume=False,
                                    render_mode="human")])

    batch_size_per_env = int(np.ceil(float(args.batch_size) / args.n_env))
    if args.net_layer == 5:
        pi_arch = [512, 512, 512, 256, 128]
        vf_arch = [512, 512, 512, 256, 128]
    elif args.net_layer == 4:
        pi_arch = [512, 512, 256, 128]
        vf_arch = [512, 512, 256, 128]
    elif args.net_layer == 6:
        pi_arch = [512, 512, 512, 256, 256, 128]
        vf_arch = [512, 512, 512, 256, 256, 128]
    elif args.net_layer == 7:
        pi_arch = [512, 512, 512, 512, 256, 256, 128]
        vf_arch = [512, 512, 512, 512, 256, 256, 128]
    elif args.net_layer == 3:
        pi_arch = [512, 256, 128]
        vf_arch = [512, 256, 128]
    policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                         net_arch=dict(pi=pi_arch, vf=vf_arch),  # changed from [512, 256] 
                         log_std_init=-2.1,)  # -2.1  for ppo19
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=args.lr,
                n_steps=batch_size_per_env,  # The number of steps to for each sub_env per actor network iteration update/rollout
                batch_size=args.minibatch,  # Minibatch size
                n_epochs=args.epoch,
                gamma=args.gamma,
                verbose=1,
                tensorboard_log=root_dir+"/../saved")

    if args.what == "train":
        if args.resume:
            # 1.resume model weights
            assert args.load_step is not None
            assert args.trial is not None
            trial = args.trial
            weight = root_dir + "/../saved/PPO_{0}/models/model_{1}_steps".format(trial, args.load_step)
            print("load: {}".format(weight))
            model = model.load(weight, print_system_info=True, env=env)

            # 2.resume env statistics
            if args.normalize_obs:
                stats_file = f'{root_dir}/../saved/PPO_{args.trial}/models/model_vecnormalize_{args.load_step}_steps.pkl'
                assert os.path.isfile(stats_file), "[Fatal]:env static file isn't exist"
                env = VecNormalize.load(stats_file, env)
                env.training = False  # do not update them at test time
                env.norm_reward = False  # reward normalization is not needed at test time

        trial = get_latest_run_id(root_dir + "/../saved", "PPO") + 1
        save_freq = batch_size_per_env*args.save_interval  # steps
        # reward_threshold_callback = RewardThresholdCallback(threshold=1500, env=env, model=model)
        checkpoint_callback = CheckpointCallback(save_freq=save_freq,  # counter for one sub_env steps
                                                 save_path=root_dir + "/../saved/PPO_{0}/models".format(trial),
                                                 name_prefix='model',
                                                 save_vecnormalize=args.normalize_obs)
        # start_randomizing_callback = StartRandomizingCallback(threshold=200, env=env, model=model)
        # start_command_callback = StartCommandCallback(threshold=100, env=env, model=model)
        callbacks = CallbackList([checkpoint_callback])
        model.learn(total_timesteps=args.max_step, callback=callbacks)

    elif args.what == "test":
        # 1. load the model parameters.
        if args.trial is not None:
            if args.load_step is not None:
                load_step = args.load_step
            else:
                tlog_path = glob.glob(root_dir + "/../saved/PPO_{0}/events.out*".format(args.trial))[0]
                tlog = EventAccumulator(tlog_path)
                tlog.Reload()

                scalars = tlog.Scalars('rollout/ep_rew_mean')
                rew_data = pd.DataFrame({"step": [s.step for s in scalars], "value": [s.value for s in scalars]})
                n_all_iter = len(rew_data.iloc[args.save_interval-1::args.save_interval])
                sorted_rew_data = (rew_data.iloc[args.save_interval-1::args.save_interval])[int(args.best_rate*n_all_iter):].sort_values(by="value")
                print(sorted_rew_data)
                load_iter = sorted_rew_data.tail(1).index[0]
                load_step = sorted_rew_data.loc[load_iter, 'step']
            weight = root_dir + "/../saved/PPO_{0}/models/model_{1}_steps".format(args.trial, load_step)
            print("load: {}".format(weight))
            model = model.load(weight, print_system_info=True)

            # load env statistics
            if args.normalize_obs:
                assert isinstance(env, VecEnv)
                stats_file = f'{root_dir}/../saved/PPO_{args.trial}/models/model_vecnormalize_{load_step}_steps.pkl'
                assert os.path.isfile(stats_file), "[Fatal]:env static file isn't exist"
                env = VecNormalize.load(stats_file, env)
                env.training = False  # do not update them at test time
                env.norm_reward = False  # reward normalization is not needed at test time
        else:
            raise ValueError("Please assign the trail number")

        # 2. simulation/evaluation
        step = 0
        start_time = time.time()
        # print(env.reset())
        # raise None
        obs = env.reset()
        for i in range(args.max_step):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            # obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            step += 1
            if args.render:
                env.render()
            if dones:
                print("step: {}".format(step))
                print("----------")
                step = 0

            end_time = time.time()
            # print("{} [sec]".format(end_time-start_time))
            if float(end_time-start_time) < env.get_attr("dt")[0]:
                if args.wait:
                    time.sleep(env.dt-float(end_time-start_time))
            start_time = end_time
    else:
        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    main()

