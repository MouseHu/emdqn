import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import tempfile
import time
import pygame
from pyvirtualdisplay import Display
import matplotlib.pyplot as plt
import sys
import readchar
from baselines.deepq.dqn_utils import *
import baselines.common.tf_util as U
import datetime
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBufferContra, PrioritizedReplayBuffer
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
from baselines.deepq.experiments.atari.model import contrastive_model, rp_model
# from baselines.deepq.experiments.atari.lru_knn_ucb import LRU_KNN_UCB
from baselines.deepq.experiments.atari.lru_knn_test import LRU_KNN_TEST
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
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e5), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(5e6),
                        help="total number of steps to run the environment for")
    parser.add_argument("--negative-samples", type=int, default=10, help="numbers for negative samples")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=16,
                        help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=10000,
                        help="number of iterations between every target network update")
    parser.add_argument("--knn", type=int, default=4, help="number of k nearest neighbours")
    parser.add_argument("--end_training", type=int, default=0, help="number of pretrain steps")
    parser.add_argument('--map_config', type=str,
                        help='The map and config you want to run in MonsterKong.',
                        default='../../../ple/configs/config_ppo_mk_large.py')
    # Bells and whistles
    # Checkpointing
    boolean_flag(parser, "rp", default=False, help="whether or not to use random projection")
    boolean_flag(parser, "learning", default=False,
                 help="if true and model was continued learned")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="latent_dim")
    parser.add_argument("--image_path", type=str, default="./images/",
                        help="image path")
    parser.add_argument("--video_path", type=str, default="./videos",
                        help="video path")
    parser.add_argument("--comment", type=str, default=datetime.datetime.now().strftime("%I-%M_%B-%d-%Y"),
                        help="discription for this experiment")

    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    monitored_env = SimpleMonitor(env)  # puts rewards and number of steps in info, before environment is wrapped
    env = wrap_dqn(
        monitored_env)  # applies a bunch of modification to simplify the observation space (downsample, make b/w)
    return env, monitored_env


args = parse_args()
# Parse savedir and azure container.
# savedir = args.save_dir
# if args.save_azure_container is not None:
#     account_name, account_key, container_name = args.save_azure_container.split(":")
#     container = Container(account_name=account_name,
#                           account_key=account_key,
#                           container_name=container_name,
#                           maybe_create=True)
#     if savedir is None:
#         # Careful! This will not get cleaned up. Docker spoils the developers.
#         savedir = tempfile.TemporaryDirectory().name
# else:
#     container = None
# Create and seed the env.
# env, monitored_env = make_env(args.env)
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

if args.seed > 0:
    set_global_seeds(args.seed)
    env.unwrapped.seed(args.seed)
print("obs shape", env.observation_space.shape)
env = GIFRecorder(video_path=args.video_path + "/{}/".format(args.comment), record_video=True, env=env)
subdir = (datetime.datetime.now()).strftime("%m-%d-%Y-%H:%M:%S") + " " + args.comment
# tf_writer = tf.summary.FileWriter(os.path.join(args.log_dir, subdir), tf.get_default_graph())
value_summary = tf.Summary()
qec_summary = tf.Summary()
value_summary.value.add(tag='discount_reward_mean')
value_summary.value.add(tag='non_discount_reward_mean')
# value_summary.value.add(tag='episode')

qec_summary.value.add(tag='qec_mean')
qec_summary.value.add(tag='qec_fount')
value_summary.value.add(tag='steps')
value_summary.value.add(tag='episodes')

with U.make_session(4) as sess:
    # EMDQN

    exploration = PiecewiseSchedule([
        (0, 1.0),
        (args.end_training, 1.0),
        # (args.end_training+1, 1.0),
        # (args.end_training+1, 0.005),
        (args.end_training + 100000, 0.005),
        # (approximate_num_iters / 5, 0.1),
        # (approximate_num_iters / 3, 0.01)
    ], outside_value=0.005)
    replay_buffer = ReplayBufferContra(args.replay_buffer_size)
    ec_buffer = []
    buffer_size = int(100000)

    # input_dim = 1024
    for i in range(env.action_space.n):
        ec_buffer.append(
            LRU_KNN_TEST(buffer_size, args.latent_dim, 'game', ob_dims=env.observation_space.shape, action=i,
                         mode=args.mode))
    # rng = np.random.RandomState(123456)  # deterministic, erase 123456 for stochastic
    # rp = rng.normal(loc=0, scale=1. / np.sqrt(latent_dim), size=(latent_dim, input_dim))
    qecwatch = []
    update_counter = 0
    qec_found = 0
    sequence = []
    num_episodes = 0
    tfout = open(
        './results/result_%s_contrast_%s' % (args.env, args.comment), 'w+')


    def act_human(ob):
        global num_iters, record_episode, start_steps, env
        z = z_func(ob)
        z = np.array(z).reshape((args.latent_dim))
        key = readchar.readkey()
        if args.env == 'MK':
            key_dict = {'s': 4, 'a': 0, 'd': 1, 'w': 3, 'x': 2, 'p': 5}
        else:
            key_dict = {'s': 5, 'a': 4, 'd': 3, 'w': 0, 'x': 2, ' ': 1, 'p': 7}
        action = key_dict.get(key, -1)
        if action == -1:
            exit(0)
        qs = np.zeros(env.action_space.n)
        counts = np.zeros(env.action_space.n)

        if record_episode:
            save_dir = os.path.join(args.image_path, "episode{}_{}".format(num_episodes, num_iters - start_steps))
        else:
            save_dir = None

        for a in range(env.action_space.n):
            # print(np.array(z).shape)
            qs[a], counts[a], find = ec_buffer[a].act_value(z, args.knn, save_dir=save_dir)
        print(qs)
        return action, z


    def option():
        global auto, record_episode
        while True:
            print("do you want to auto run the agent for the next n episode? please input n")
            x = input()
            try:
                auto = int(x)
                break
            except:
                print("not an integer!")
                pass
        while True:
            print("do you want to record images? r for yes, n for no")
            key = readchar.readkey()
            print(key)
            if key == 'r':
                record_episode = True
                break
            elif key == 'n':
                record_episode = False
                break
            else:
                pass


    def render():
        global env
        # plt.imshow(env.render(mode='rgb_array'))
        env.render()


    def act(ob, stochastic=0, update_eps=-1):
        global eps, qecwatch, qec_found, num_iters, record_episode, start_steps
        # print(ob.shape)
        z = z_func(ob)
        z = np.array(z).reshape((args.latent_dim))
        if update_eps >= 0:
            eps = update_eps
        if np.random.random() < max(stochastic, eps):
            action = np.random.randint(0, env.action_space.n)
            # print(eps,env.action_space.n,action)
            return action, z
        else:
            # print(eps,stochastic,np.random.rand(0, 1))
            qs = np.zeros(env.action_space.n)
            counts = np.zeros(env.action_space.n)
            if record_episode:
                save_dir = os.path.join(args.image_path, "episode{}_{}/".format(num_episodes, num_iters - start_steps))
            else:
                save_dir = None
            for a in range(env.action_space.n):
                # print(np.array(z).shape)
                qs[a], counts[a], find = ec_buffer[a].act_value(z, args.knn, save_dir)
                if find:
                    qecwatch.append(qs[a])
                    qec_found += 1

            optimistic_q = qs
            # print(qs)
            q_max = np.max(optimistic_q)
            # print("optimistic q", optimistic_q.shape, np.where(optimistic_q == q_max))
            max_action = np.where(optimistic_q == q_max)[0]
            # print(max_action)
            action_selected = np.random.randint(0, len(max_action))
            # print("ec",eps,np.argmax(q),q)
            return max_action[action_selected], z


    def update_kdtree():
        for a in range(env.action_space.n):
            ec_buffer[a].update_kdtree()


    def update_ec(sequence):
        _, _, acts, _ = list(zip(*sequence))
        # print(np.bincount(acts))
        Rtd = 0.
        Rtds = [0]
        for seq in reversed(sequence):
            s, z, a, r = seq
            # z = s.flatten()
            # z = np.dot(rp, s.flatten())
            Rtd = r + args.gamma * Rtd
            Rtds.append(Rtd)
            z = np.array(z).reshape((args.latent_dim))
            qd, _ = ec_buffer[a].peek(z, Rtd, True)
            if qd == None:  # new action
                ec_buffer[a].add(z, Rtd, s)
        return Rtds


    # Create training graph and replay buffer
    z_func, train = deepq.build_train_contrast(
        make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
        model_func=rp_model if args.rp else contrastive_model,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
        gamma=args.gamma,
        grad_norm_clipping=10,
    )

    U.initialize()
    num_iters = 0
    num_episodes = 0
    non_discount_return = [0.0]
    discount_return = [0.0]
    # Load the model
    # state = maybe_load_model(savedir, container)
    # if state is not None:
    #     num_iters, replay_buffer = state["num_iters"], state["replay_buffer"],
    # monitored_env.set_state(state["monitor_state"])

    obs = env.reset()
    # Main trianing loop
    auto = 0
    start_steps = 0
    record_episode = True
    option()
    display = Display(visible=1, size=(640, 480))
    display.start()
    while True:
        num_iters += 1
        # Take action and store transition in the replay buffer.
        if auto == 0:
            action, z = act_human(np.array(obs)[None])
        else:
            action, z = act(np.array(obs)[None], update_eps=exploration.value(num_iters))

        new_obs, rew, done, info = env.step(action)
        if args.env != "MK":
            render()
        non_discount_return[-1] += rew
        discount_return[-1] += rew * args.gamma ** (num_iters - start_steps)
        # EMDQN
        sequence.append([obs, z, action, np.clip(rew, -1, 1)])
        # replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs
        if done:
            # print((num_iters - start_steps), args.gamma ** (num_iters - start_steps))
            if num_iters >= args.end_training and auto > 0:
                update_ec(sequence)
            print("end of episode {},iters {},auto remain {}".format(num_episodes, num_iters,auto))
            num_episodes += 1
            auto -= 1
            if auto <= 0:
                option()
            # EMDQN
            # num_episodes += 1
            obs = env.reset()
            non_discount_return.append(0.0)
            discount_return.append(0.0)
            start_steps = num_iters

        # if num_iters % args.learning_freq == 0 and len(replay_buffer) > args.batch_size * (
        #         args.negative_samples + 1) and num_iters < args.end_training and args.learning:
        #     # train vae
        #     obses_t, actions, rewards, obses_tp1, dones, obses_neg = replay_buffer.sample(args.batch_size)
        #     inputs = [[1], obses_t, obses_tp1, obses_neg]
        #     total_errors, summary = train(*inputs)
        # tf_writer.add_summary(summary, global_step=num_iters)
        # tf_writer.add_summary(summary,global_step=info["steps"])
        # Update target network.
        if num_iters % args.target_update_freq == 0 and num_iters > args.end_training:  # NOTE: why not 10000?
            update_kdtree()

        start_time = time.time()
