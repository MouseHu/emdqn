import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import tempfile
import time

import sys

cwd = os.getcwd()
cwd = '/'.join(cwd.split('/')[:-4])
temp = sys.path
temp.append('')
temp[1:] = temp[0:-1]
temp[0] = cwd
print(sys.path)

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
from baselines.deepq.experiments.atari.model import contrastive_model, rp_model, modelbased_model
# from baselines.deepq.experiments.atari.lru_knn_ucb import LRU_KNN_UCB
from baselines.deepq.experiments.atari.lru_knn_ucb import LRU_KNN_UCB
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
    parser.add_argument("--knn", type=int, default=11, help="number of k nearest neighbours")
    parser.add_argument("--end_training", type=int, default=2e5, help="number of pretrain steps")
    parser.add_argument('--map_config', type=str,
                        help='The map and config you want to run in MonsterKong.',
                        default='../../../ple/configs/config_ppo_mk_large.py')
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


def maybe_save_model(savedir, container, state):
    """This function checkpoints the model and state of the training algorithm."""
    if savedir is None:
        return
    start_time = time.time()
    model_dir = "model-{}".format(state["num_iters"])
    U.save_state(os.path.join(savedir, model_dir, "saved"))
    if container is not None:
        container.put(os.path.join(savedir, model_dir), model_dir)
    relatively_safe_pickle_dump(state, os.path.join(savedir, 'training_state.pkl.zip'), compression=True)
    if container is not None:
        container.put(os.path.join(savedir, 'training_state.pkl.zip'), 'training_state.pkl.zip')
    relatively_safe_pickle_dump(state["monitor_state"], os.path.join(savedir, 'monitor_state.pkl'))
    if container is not None:
        container.put(os.path.join(savedir, 'monitor_state.pkl'), 'monitor_state.pkl')
    logger.log("Saved model in {} seconds\n".format(time.time() - start_time))


def maybe_load_model(savedir, container):
    """Load model if present at the specified path."""
    if savedir is None:
        return

    state_path = os.path.join(os.path.join(savedir, 'training_state.pkl.zip'))
    if container is not None:
        logger.log("Attempting to download model from Azure")
        found_model = container.get(savedir, 'training_state.pkl.zip')
    else:
        found_model = os.path.exists(state_path)
    if found_model:
        state = pickle_load(state_path, compression=True)
        model_dir = "model-{}".format(state["num_iters"])
        if container is not None:
            container.get(savedir, model_dir)
        U.load_state(os.path.join(savedir, model_dir, "saved"))
        logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
        return state


if __name__ == '__main__':
    args = parse_args()
    if args.train_latent:
        print("Training latent")
    # Parse savedir and azure container.
    savedir = args.save_dir
    if args.save_azure_container is not None:
        account_name, account_key, container_name = args.save_azure_container.split(":")
        container = Container(account_name=account_name,
                              account_key=account_key,
                              container_name=container_name,
                              maybe_create=True)
        if savedir is None:
            # Careful! This will not get cleaned up. Docker spoils the developers.
            savedir = tempfile.TemporaryDirectory().name
    else:
        container = None
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
    # env = GIFRecorder(video_path=args.video_path + "/{}/".format(args.comment), record_video=True, env=env)
    subdir = (datetime.datetime.now()).strftime("%m-%d-%Y-%H:%M:%S") + " " + args.comment
    tf_writer = tf.summary.FileWriter(os.path.join(args.log_dir, subdir), tf.get_default_graph())
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
        buffer_size = 1000000
        ec_buffer = LRU_KNN_UCB(buffer_size, args.latent_dim, 'game', mode=args.mode)

        # rng = np.random.RandomState(123456)  # deterministic, erase 123456 for stochastic
        # rp = rng.normal(loc=0, scale=1. / np.sqrt(latent_dim), size=(latent_dim, input_dim))
        qecwatch = []
        update_counter = 0
        qec_found = 0
        sequence = []

        tfout = open(
            './results/result_%s_contrast_%s' % (args.env, args.comment), 'w+')


        def act(ob, stochastic=0, update_eps=-1):
            global eps, qecwatch, qec_found, num_iters
            # print(ob.shape)

            z = z_func(ob)
            # next_z, rs = model_func(np.tile(ob, [env.action_space.n, 1, 1, 1]), actions)
            z = np.array(z).reshape((args.latent_dim))
            if update_eps >= 0:
                eps = update_eps
            if np.random.random() < max(stochastic, eps):
                action = np.random.randint(0, env.action_space.n)
                # print(eps,env.action_space.n,action)
                return action, z
            else:
                # print(eps,stochastic,np.random.rand(0, 1))
                # qs = np.zeros(env.action_space.n)
                actions = np.arange(env.action_space.n)
                next_zs, rs = model_func(np.tile(z, [env.action_space.n, 1]), actions)
                vs = ec_buffer.knn_value(next_zs[0], args.knn)
                qs = args.gamma * np.array(vs) + np.array(rs)
                optimistic_q = qs
                q_max = np.max(optimistic_q)
                # print("optimistic q", optimistic_q.shape, np.where(optimistic_q == q_max))
                max_action = np.where(optimistic_q == q_max)[0]
                # print(max_action)
                action_selected = np.random.randint(0, len(max_action))
                # print("ec",eps,np.argmax(q),q)
                return max_action[action_selected], z


        def update_kdtree():
            ec_buffer.update_kdtree()


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
                v, _ = ec_buffer.peek(z, Rtd, True)
                if v == None:  # new action
                    ec_buffer.add(z, Rtd)
            return Rtds


        # Create training graph and replay buffer
        z_func, model_func, train = deepq.build_train_modelbased(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            net_func=rp_model if args.rp else contrastive_model,
            model_func=modelbased_model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
            gamma=args.gamma,
            grad_norm_clipping=10,
        )

        approximate_num_iters = args.num_steps
        if args.ucb:
            exploration = PiecewiseSchedule([
                (0, 1),
                (2e4, 1),
            ], outside_value=0.01)
        else:
            exploration = PiecewiseSchedule([
                (0, 1.0),
                (args.end_training, 1.0),
                # (args.end_training+1, 1.0),
                # (args.end_training+1, 0.005),
                (args.end_training + 100000, 0.01),
                # (approximate_num_iters / 5, 0.1),
                # (approximate_num_iters / 3, 0.01)
            ], outside_value=0.01)

        replay_buffer = ReplayBufferContra(args.replay_buffer_size, K=args.negative_samples)

        U.initialize()
        num_iters = 0
        num_episodes = 0
        non_discount_return = [0.0]
        discount_return = [0.0]
        # Load the model
        state = maybe_load_model(savedir, container)
        # if state is not None:
        #     num_iters, replay_buffer = state["num_iters"], state["replay_buffer"],
        # monitored_env.set_state(state["monitor_state"])

        start_time, start_steps = time.time(), 0
        steps_per_iter = RunningAvg(0.999)
        iteration_time_est = RunningAvg(0.999)
        obs = env.reset()
        print_flag = True
        # Main trianing loop
        train_time = 0
        act_time = 0
        env_time = 0
        update_time = 0
        cur_time = time.time()
        while True:
            num_iters += 1
            # Take action and store transition in the replay buffer.
            action, z = \
                act(np.array(obs)[None], update_eps=exploration.value(num_iters))
            act_time += time.time() - cur_time
            cur_time = time.time()
            new_obs, rew, done, info = env.step(action)
            env_time += time.time() - cur_time
            cur_time = time.time()
            # if num_episodes % 40 == 39:
            #     env.record = True
            non_discount_return[-1] += rew
            discount_return[-1] += rew * args.gamma ** (num_iters - start_steps)
            # EMDQN
            sequence.append([obs, z, action, np.clip(rew, -1, 1)])
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs
            if done:
                # print((num_iters - start_steps), args.gamma ** (num_iters - start_steps))
                num_episodes += 1
                # EMDQN
                if num_iters >= args.end_training:
                    update_ec(sequence)
                    update_time += time.time() - cur_time
                    cur_time = time.time()

                if print_flag:
                    print(info)
                    print_flag = False

                obs = env.reset()
                non_discount_return.append(0.0)
                discount_return.append(0.0)

            if num_iters % args.learning_freq == 0 and len(replay_buffer) > args.batch_size * (
                    args.negative_samples + 1) and (
                    num_iters < args.end_training or args.learning):
                # train vae
                obses_t, actions, rewards, obses_tp1, dones, obses_neg = replay_buffer.sample(args.batch_size, False)
                obses_t_c, actions_c, rewards_c, obses_tp1_c, dones_c, obses_neg_c = replay_buffer.sample(args.batch_size)
                inputs = [[1],obses_t_c,obses_tp1_c,obses_neg_c,obses_t, obses_tp1, rewards,actions]
                # inputs = [obses_t, obses_tp1, rewards, actions]
                total_errors, summary = train(*inputs)
                tf_writer.add_summary(summary, global_step=num_iters)
                # tf_writer.add_summary(summary,global_step=info["steps"])
                # Update target network.
            train_time += time.time() - cur_time
            cur_time = time.time()
            if num_iters % args.target_update_freq == 0 and num_iters > args.end_training:  # NOTE: why not 10000?
                update_kdtree()
            if start_time is not None:
                steps_per_iter.update(1)
                iteration_time_est.update(time.time() - start_time)
            start_time = time.time()
            value_summary.value[2].simple_value = num_iters

            # Save the model and training state.
            '''
            if num_iters > 0 and (num_iters % args.save_freq == 0 or info["steps"] > args.num_steps):
                maybe_save_model(savedir, container, {
                    'replay_buffer': replay_buffer,
                    'num_iters': num_iters,
                    'monitor_state': monitored_env.get_state()
                })
            '''

            if num_iters > args.num_steps:
                break

            if done:
                return_len = min(len(non_discount_return) - 1, 100)
                sequence = []
                steps_left = args.num_steps - num_iters
                completion = np.round(num_iters / args.num_steps, 2)

                logger.record_tabular("% completion", completion)
                # logger.record_tabular("steps", info["steps"])
                logger.record_tabular("iters", num_iters)
                # logger.record_tabular("episodes", info[0]["episode"])
                logger.record_tabular("reward", np.mean(non_discount_return[-return_len - 1:-1]))
                logger.record_tabular("discount reward", np.mean(discount_return[-return_len - 1:-1]))
                logger.record_tabular("num episode", num_episodes)
                # logger.record_tabular("qec_mean", np.mean(qecwatch))
                # logger.record_tabular("qec_proportion", qec_found / (num_iters - start_steps))
                logger.record_tabular("update time", update_time)
                logger.record_tabular("train time", train_time)
                logger.record_tabular("act_time", act_time)
                logger.record_tabular("env_time", env_time)
                value_summary.value[0].simple_value = np.mean(discount_return[-return_len - 1:-1])
                value_summary.value[1].simple_value = np.mean(non_discount_return[-return_len - 1:-1])
                value_summary.value[3].simple_value = num_episodes
                # qec_summary.value[0].simple_value = np.mean(qecwatch)
                # qec_summary.value[1].simple_value = qec_found / (num_iters - start_steps)

                # if return_len > 1:
                #     # np.mean(np.mean(episodic_return[-return_mean + 1:-1]))
                #     tfout.write("%d, %.2f\n" % (num_iters, int(np.mean(discount_return[-return_len - 1:-1]))))
                #     tfout.flush()
                logger.record_tabular("exploration", exploration.value(num_iters))
                fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                                if steps_per_iter._value is not None else 1 / (float(iteration_time_est) + 1e-6))
                logger.dump_tabular()
                logger.log()
                logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                logger.log()

                start_steps = num_iters
                # qecwatch = []
                # qec_found = 0
            total_steps = num_iters - args.end_training
            tf_writer.add_summary(value_summary, global_step=total_steps)
            # tf_writer.add_summary(qec_summary, global_step=total_steps)
            cur_time = time.time()
