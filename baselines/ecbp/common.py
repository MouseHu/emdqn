import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import tempfile
import time

import sys

from baselines.deepq.dqn_utils import *
import baselines.common.tf_util as U
import datetime
from baselines import logger
from baselines import deepq
from baselines.common.atari_wrappers_deprecated import FrameStack
from baselines.deepq.replay_buffer import ReplayBufferHash, PrioritizedReplayBuffer
from baselines.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
    SimpleMonitor
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
# when updating this to non-deperecated ones, it is important to
# copy over LazyFrames
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from baselines.common.azure_utils import Container
from baselines.deepq.experiments.atari.model import contrastive_model, rp_model, model

# from baselines.deepq.experiments.atari.lru_knn_count_gpu_fixmem import LRU_KNN_COUNT_GPU_FIXMEM
from baselines.deepq.experiments.atari.lru_knn_combine_bp import LRU_KNN_COMBINE_BP
from baselines.common.atari_lib import create_atari_environment


# from gym.wrappers.monitoring.video_recorder import VideoRecorder


def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment test deployment
    parser.add_argument("--env", type=str, default="Pong", help="name of the game")
    parser.add_argument("--seed", type=int, default=int(time.time()), help="which seed to use")
    parser.add_argument("--gamma", type=int, default=0.99, help="which seed to use")
    # Core DQN parameters
    parser.add_argument("--mode", type=str, default="max", help="mode of episodic memory")
    parser.add_argument("--buffer-size", type=int, default=int(1e5), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning6 rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(5e6),
                        help="total number of steps to run the environment for")
    parser.add_argument("--negative-samples", type=int, default=10, help="numbers for negative samples")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=16,
                        help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=5000,
                        help="number of iterations between every target network update")
    parser.add_argument("--knn", type=int, default=4, help="number of k nearest neighbours")
    parser.add_argument("--end_training", type=int, default=0, help="number of pretrain steps")
    parser.add_argument('--map_config', type=str,
                        help='The map and config you want to run in MonsterKong.',
                        default='../ple/configs/config_ppo_mk_hard_2.py')
    # Bells and whistles
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default=None,
                        help="directory in which training state and model should be saved.")
    parser.add_argument("--save-azure-container", type=str, default=None,
                        help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
    parser.add_argument("--save-freq", type=int, default=1e6,
                        help="save model once every time this many iterations are completed")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="latent_dim")
    parser.add_argument("--video_path", type=str, default="./videos",
                        help="video path")
    parser.add_argument("--comment", type=str, default=datetime.datetime.now().strftime("%I-%M_%B-%d-%Y"),
                        help="discription for this experiment")
    parser.add_argument("--log_dir", type=str, default="./tflogs",
                        help="directory in which training state and model should be saved.")
    boolean_flag(parser, "load-on-start", default=True,
                 help="if true and model was previously saved then training will be resumed")

    boolean_flag(parser, "learning", default=False,
                 help="if true and model was continued learned")

    boolean_flag(parser, "exploration", default=False,
                 help="if true and model was continued learned")

    boolean_flag(parser, "ucb", default=False, help="whether or not to use ucb exploration")
    boolean_flag(parser, "dueling", default=True, help="whether or not to use dueling")
    boolean_flag(parser, "bp", default=True, help="whether or not to use bp")
    boolean_flag(parser, "episodic", default=True, help="whether or not to use dueling")
    boolean_flag(parser, "baseline", default=False, help="if baseline use episodic memory instead of network")
    boolean_flag(parser, "imitate", default=False, help="if baseline use episodic memory instead of network")
    boolean_flag(parser, "rp", default=False, help="whether or not to use random projection")
    # EMDQN
    boolean_flag(parser, "train-latent", default=False, help="whether or not to further train latent")
    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    monitored_env = SimpleMonitor(env)  # puts rewards and number of steps in info, before environment is wrapped
    env = wrap_dqn(
        monitored_env)  # applies a bunch of modification to simplify the observation space (downsample, make b/w)
    return env, monitored_env


def create_env(args):
    if args.env == "MK":
        import imp

        try:
            map_config_file = args.map_config
            map_config = imp.load_source('map_config', map_config_file).map_config
        except Exception as e:
            sys.exit(str(e) + '\n'
                     + 'map_config import error. File not exist or map_config not specified')
        from gym.envs.registration import register

        register(
            id='MonsterKong-v0',
            entry_point='baselines.ple.gym_env.monsterkong:MonsterKongEnv',
            kwargs={'map_config': map_config},
        )

        env = gym.make('MonsterKong-v0')
        env = ProcessFrame(env)
    else:
        env = create_atari_environment(args.env, sticky_actions=False)
        env = FrameStack(env, 4)
    if args.seed > 0:
        set_global_seeds(args.seed)
        env.unwrapped.seed(args.seed)
    return env