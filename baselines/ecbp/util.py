import argparse
import time
import sys
import yaml
from baselines.deepq.dqn_utils import *
import baselines.common.tf_util as U
import datetime
from baselines import logger
from baselines import deepq
from baselines.ecbp.env.d4rl_wrapper import D4RLDiscreteMazeEnvWrapper
from baselines.ecbp.env.fourrooms import Fourrooms
# from baselines.ecbp.env.tworooms import Tworooms
from baselines.common.atari_wrappers_deprecated import FrameStack
from baselines.common.atari_lib import MKPreprocessing
from baselines.common.atari_lib import CropWrapper
from baselines.common.atari_lib import NoisyEnv
from baselines.common.atari_lib import DoomPreprocessing
# from baselines.doom.environment import DoomEnvironment
from baselines.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
    SimpleMonitor
)
from baselines.atari.environment import Environment as atari_env_vast
from baselines.common.atari_lib import create_atari_environment
from baselines.ecbp.agents.ecbp_agent import ECBPAgent
from baselines.ecbp.agents.ps_agent import PSAgent
from baselines.ecbp.agents.ps_mp_agent import PSMPAgent
from baselines.ecbp.agents.ps_mp_learning_agent import PSMPLearnAgent
from baselines.ecbp.agents.psmp_learning_target_agent import PSMPLearnTargetAgent
from baselines.ecbp.agents.kbps_mp_agent import KBPSMPAgent
from baselines.ecbp.agents.kbps_agent import KBPSAgent
from baselines.ecbp.agents.ec_agent import ECAgent
from baselines.ecbp.agents.human_agent import HumanAgent
from baselines.ecbp.agents.hybrid_agent import HybridAgent, HybridAgent2
from baselines.ecbp.agents.graph.model import representation_model_cnn, representation_model_mlp, rp_model, \
    contrastive_model
import logging

mk_map_config = {"small": "../ple/configs/config_ppo_mk.py", "hard": "../ple/configs/config_ppo_mk_hard.py",
                 "hard_2": "../ple/configs/config_ppo_mk_hard_2.py"}


# from gym.wrappers.monitoring.video_recorder import VideoRecorder


def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment test deployment
    parser.add_argument("--env", type=str, default="Atari", help="name of the game")
    parser.add_argument("--env_name", type=str, default="Pong", help="name of the game")
    parser.add_argument("--seed", type=int, default=int(time.time()), help="which seed to use")
    parser.add_argument("--gamma", type=int, default=0.99, help="which seed to use")
    # Core DQN parameters
    parser.add_argument("--mode", type=str, default="max", help="mode of episodic memory")
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="replay buffer size")
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
    parser.add_argument("--noise_dim", type=int, default=4, help="number of noisy dim")
    parser.add_argument("--noise_var", type=float, default=1, help="number of noisy var")
    parser.add_argument("--end_training", type=int, default=0, help="number of pretrain steps")
    parser.add_argument("--eval_epsilon", type=int, default=0.01, help="eval epsilon")
    parser.add_argument("--queue_threshold", type=int, default=1e-7, help="queue_threshold")
    parser.add_argument('--map_config', type=str,
                        help='The map and config you want to run in MonsterKong.',
                        default='../ple/configs/config_ppo_mk_hard.py')

    parser.add_argument('--param_dir', type=str,
                        help='The map and config you want to run in MonsterKong.',
                        default='../doom/')
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
    parser.add_argument("--base_log_dir", type=str, default="/data1/hh/ecbp",
                        help="directory in which training state and model should be saved.")
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
    boolean_flag(parser, "debug", default=False, help="whether or not to output detail info")
    boolean_flag(parser, "vector_input", default=False, help="if env is vector input")
    boolean_flag(parser, "render", default=False, help="if render env")
    # EMDQN
    boolean_flag(parser, "train-latent", default=False, help="whether or not to further train latent")
    return parser.parse_args()


def load_params(subdir, experiment):
    with open("%s/params.yaml" % subdir, 'rb') as stream:
        params = yaml.load(stream)
    experiment_params = params['env_params']['experiments'][experiment]
    for (key, val) in experiment_params.items():
        params['env_params'][key] = val
    params['env_params'].pop('experiments')
    for (key, val) in params.items():
        if (not "_params" in key) and (key != 'net_arches'):
            params['env_params'][key] = val
            params['model_params'][key] = val
            params['agent_params'][key] = val
    return params


def create_env(args):
    if args.env == "MK" or args.env == "mk":
        import imp

        try:
            map_config_file = mk_map_config.get(args.env_name, mk_map_config["small"])
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
        env = MKPreprocessing(env, frame_skip=3, no_jump=True)
    elif args.env == "GW" or args.env == "gw":
        if args.env_name == "fourrooms":
            env = Fourrooms()
        else:
            env = Tworooms()
    elif args.env == "doom" or args.env == "Doom":
        params = load_params(args.param_dir, args.env_name)
        env = DoomEnvironment(**params['env_params'])
        env = DoomPreprocessing(env, frame_skip=4)
    elif args.env == "atari" or args.env == "Atari":
        env = create_atari_environment(args.env_name, sticky_actions=False)
        env = FrameStack(env, 4)
        # if args.env_name == "Pong":
        #     env = CropWrapper(env, 34, 15)
    elif args.env == "vast":

        params = load_params("../atari/", args.env_name)
        params["env_params"]['game'] = args.env_name
        env = atari_env_vast(**params["env_params"])
    elif args.env == "mujoco":
        from gym.envs.registration import register
        goal_args = [[8.0, 0.0], [8 + 1e-3, 0 + 1e-3]]
        random_start = False
        # The episode length for test is 500
        max_timestep = 500

        register(
            id='PointMazeTest-v10',
            entry_point='mujoco.create_maze_env:create_maze_env',
            kwargs={'env_name': 'DiscretePointMaze', 'goal_args': goal_args, 'maze_size_scaling': 4,
                    'random_start': random_start},
            max_episode_steps=max_timestep,
        )
        env = gym.make('PointMazeTest-v10')
    elif args.env == "noise_atari" or args.env == "atari_noise":
        game_version = 'v0'
        env = gym.make('{}-{}'.format(args.env_name, game_version))
        env = NoisyEnv(env, args.noise_dim, args.noise_var)
    elif args.env == "d4rl":
        import d4rl
        env = gym.make(args.env_name)

        if "maze" in args.env_name:
            env = D4RLDiscreteMazeEnvWrapper(env)

    else:
        raise NotImplementedError
    if args.seed > 0:
        set_global_seeds(args.seed)
        env.unwrapped.seed(args.seed)
    return env


def make_logger(name, filename, stream_level=logging.INFO, file_level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(filename)
    fh.setLevel(file_level)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(stream_level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


def make_agent(args, env, tf_writer):
    exploration = PiecewiseSchedule([
        (0, 1),
        (args.end_training, 1.0),
        (args.end_training + 500000, 0.05),
        (args.end_training + 1000000, 0.01),
    ], outside_value=0.01)
    agent_dict = {"ECBP": ECBPAgent, "PS": PSAgent, "PSMP": PSMPAgent, "PSMPLearn": PSMPLearnAgent, "KBPS": KBPSAgent,
                  "KBPSMP": KBPSMPAgent, "Hybrid": HybridAgent2, "Human": HumanAgent}
    agent_func = agent_dict[args.agent]
    try:
        num_actions = env.action_space.n
    except AttributeError:
        num_actions = env.unwrapped.pseudo_action_space.n
    obs_shape = env.observation_space.shape
    if obs_shape is None or obs_shape == (None,):
        obs_shape = env.unwrapped.observation_space.shape
    # print(env.unwrapped.observation_space.shape,"here!")
    input_type = U.Float32Input if args.vector_input else U.Uint8Input
    agent = agent_func(representation_model_mlp if args.rp else representation_model_cnn, exploration,
                       obs_shape, input_type,
                       args.lr,
                       args.buffer_size, num_actions, args.latent_dim, args.gamma, args.knn,
                       args.eval_epsilon, args.queue_threshold, args.batch_size,
                       tf_writer)
    return agent
