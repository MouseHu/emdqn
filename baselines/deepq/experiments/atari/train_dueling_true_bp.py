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
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e5), help="replay buffer size")
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
                        default='../../../ple/configs/config_ppo_mk_hard_2.py')
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
        env = FrameStack(env, 4)
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

    qec_summary.value.add(tag='qec_find_t')
    qec_summary.value.add(tag='qec_find_tp1')
    value_summary.value.add(tag='steps')
    value_summary.value.add(tag='episodes')

    with U.make_session(4) as sess:
        # EMDQN

        buffer_size = 1000000
        # input_dim = 1024
        ec_buffer = LRU_KNN_COMBINE_BP(env.action_space.n, buffer_size, args.latent_dim, args.latent_dim, args.gamma)
        # rng = np.random.RandomState(123456)  # deterministic, erase 123456 for stochastic
        # rp = rng.normal(loc=0, scale=1. / np.sqrt(latent_dim), size=(latent_dim, input_dim))
        qecwatch = []
        update_counter = 0
        qec_found = 0
        sequence = []


        tfout = open(
            './results/result_%s_contrast_%s' % (args.env, args.comment), 'w+')



        def act_baseline(z, stochastic=0, update_eps=-1):
            global eps, qecwatch, qec_found, num_iters
            z = np.array(z).reshape((args.latent_dim))
            if update_eps >= 0:
                eps = update_eps
            if np.random.random() < max(stochastic, eps):
                acts = np.random.randint(0, env.action_space.n)
                return acts
            else:
                qs = np.zeros((env.action_space.n, 1))
                finds = np.zeros((1,))
                for a in range(env.action_space.n):
                    qs[a], _, find = ec_buffer.act_value(np.array([z]), a, args.knn)
                    finds += sum(find)

                q_max = np.max(qs)
                max_action = np.where(qs == q_max)[0]
                action_selected = np.random.randint(0, len(max_action))
                return max_action[action_selected]


        def state_value(zs):
            batch_size = len(zs)
            qs = np.zeros((env.action_space.n, batch_size))
            inrs = np.zeros((env.action_space.n, batch_size))
            finds = np.zeros((batch_size,))
            for a in range(env.action_space.n):
                qs[a], inrs[a], find = ec_buffer.act_value(zs, a, args.knn)
                finds += sum(find)
            qs = np.transpose(qs)
            q_max = np.max(qs, axis=1)
            actions_onehot = []
            for i in range(batch_size):
                max_actions = np.where(qs[i] == q_max[i])[0]
                action_selected = np.random.randint(0, len(max_actions))
                action_onehot = np.zeros((env.action_space.n,))
                action_onehot[max_actions[action_selected]] = 1
                actions_onehot.append(action_onehot)

            return q_max, np.transpose(inrs), np.array(actions_onehot), finds


        def update_kdtree():
            pass


        def update_ec(sequence):
            ec_buffer.update_sequence(sequence,args.gamma)


        # Create training graph and replay buffer
        act, train, update_target = deepq.build_train_dueling_true(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            model_func=rp_model if args.rp else contrastive_model,
            q_func=model,
            imitate=args.imitate,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
            gamma=args.gamma,
            grad_norm_clipping=10,
        )

        approximate_num_iters = args.num_steps

        exploration = PiecewiseSchedule([
            (0, 1),
            (args.end_training, 1.0),
            # (args.end_training+1, 1.0),
            # (args.end_training+1, 0.005),
            (args.end_training + 10000, 1.0),
            (args.end_training + 200000, 0.05),
            (args.end_training + 400000, 0.01),
            # (approximate_num_iters / 5, 0.1),
            # (approximate_num_iters / 3, 0.01)
        ], outside_value=0.01)

        replay_buffer = ReplayBufferHash(args.replay_buffer_size)

        U.initialize()
        num_iters = 0
        num_episodes = 0
        non_discount_return = [0.0]
        discount_return = [0.0]
        # Load the model
        state = maybe_load_model(savedir, container)
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
            if args.baseline:
                action = act_baseline(z, update_eps=exploration.value(num_iters))
            act_time += time.time() - cur_time
            cur_time = time.time()
            new_obs, rew, done, info = env.step(action)
            _, z_tp1 = \
                act(np.array(new_obs)[None], update_eps=exploration.value(num_iters))
            env_time += time.time() - cur_time
            cur_time = time.time()
            # if num_episodes % 40 == 39:
            #     env.record = True
            non_discount_return[-1] += rew
            discount_return[-1] += rew * args.gamma ** (num_iters - start_steps)
            # EMDQN
            sequence.append([obs, z, action, np.clip(rew, -1, 1)])
            replay_buffer.add(obs, np.squeeze(z), action, rew, new_obs, np.squeeze(z_tp1), float(done))
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
                    args.negative_samples + 1) and not args.baseline:
                # train vae
                obses_t, hashes_t, actions_t, rewards, obses_tp1, hashes_tp1, dones = replay_buffer.sample(
                    args.batch_size)
                # info = [ec_buffer.act_value(h, args.knn) for h in hashes_t]
                # test = tuple(zip(*state_value(hashes_t)))
                # print(test,len(test))
                values_t, inrs_t, imit_actions_t, find_t = state_value(hashes_t)
                values_tp1, inrs_tp1, imit_actions_tp1, find_tp1 = state_value(hashes_tp1)
                # values_tp1, actions_tp1, find_tp1 = tuple(zip(*[state_value(h) for h in hashes_tp1]))
                qec_summary.value[0].simple_value = np.mean(find_t)
                qec_summary.value[1].simple_value = np.mean(find_tp1)
                # print(actions_t.shape,inrs_t.shape)
                rewards += inrs_t[np.arange(args.batch_size, dtype=np.int), np.squeeze(actions_t)]
                # actions_t = [np.array(x[1]) if x[1] is not None else np.ones(
                #     (env.action_space.n,)) / env.action_space.n for x in info]
                # print(actions_t)
                # print(np.array(actions_t).shape)
                # values_tp1 = [ec_buffer.act_value(h, args.knn)[0] for h in hashes_tp1]
                inputs = [obses_t, obses_tp1, np.array(actions_t).squeeze(), rewards, dones, np.ones_like(dones),
                          values_t,
                          values_tp1]
                if args.imitate:
                    inputs.append(imit_actions_t)
                total_errors, summary = train(*inputs)
                tf_writer.add_summary(summary, global_step=num_iters)
                tf_writer.add_summary(qec_summary, global_step=total_steps)
                # tf_writer.add_summary(summary,global_step=info["steps"])
                # Update target network.
            train_time += time.time() - cur_time
            cur_time = time.time()
            if num_iters % args.target_update_freq == 0 and num_iters > args.end_training:  # NOTE: why not 10000?
                update_kdtree()
                update_target()
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
                logger.record_tabular("qec_mean", np.mean(qecwatch))
                logger.record_tabular("qec_proportion", qec_found / (num_iters - start_steps))
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
                qecwatch = []
                qec_found = 0
            total_steps = num_iters - args.end_training
            tf_writer.add_summary(value_summary, global_step=total_steps)
            tf_writer.add_summary(qec_summary, global_step=total_steps)
            cur_time = time.time()