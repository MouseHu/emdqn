import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import tempfile
import time
import datetime
import sys

cwd = os.getcwd()
cwd = '/'.join(cwd.split('/')[:-4])
temp = sys.path
temp.append('')
temp[1:] = temp[0:-1]
temp[0] = cwd
print(sys.path)

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
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
from baselines.deepq.dqn_utils import *
# when updating this to non-deperecated ones, it is important to
# copy over LazyFrames
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from baselines.common.azure_utils import Container
# from model import
from baselines.deepq.experiments.atari.model import model, dueling_model
from baselines.deepq.experiments.atari.lru_knn import LRU_KNN
# from lru_knn import LRU_KNN


def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default="Pong", help="name of the game")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e5), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(5e6),
                        help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=4,
                        help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=40000,
                        help="number of iterations between every target network update")
    # Bells and whistles
    boolean_flag(parser, "double-q", default=False, help="whether or not to use double q learning")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    boolean_flag(parser, "prioritized", default=False, help="whether or not to use prioritized replay buffer")
    parser.add_argument("--prioritized-alpha", type=float, default=0.6,
                        help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized-beta0", type=float, default=0.4,
                        help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps", type=float, default=1e-6,
                        help="eps parameter for prioritized replay buffer")
    parser.add_argument("--comment", type=str, default=datetime.datetime.now().strftime("%I-%M_%B-%d-%Y"),
                        help="discription for this experiment")
    # Checkpointing
    parser.add_argument("--log_dir", type=str, default="./tflogs",
                        help="directory in which training state and model should be saved.")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="directory in which training state and model should be saved.")
    parser.add_argument("--save-azure-container", type=str, default=None,
                        help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
    parser.add_argument("--save-freq", type=int, default=1e6,
                        help="save model once every time this many iterations are completed")
    boolean_flag(parser, "load-on-start", default=True,
                 help="if true and model was previously saved then training will be resumed")
    parser.add_argument('--map_config', type=str,
                        help='The map and config you want to run in MonsterKong.',
                        default='../../../ple/configs/config_ppo_mk_large.py')
    # EMDQN
    boolean_flag(parser, "emdqn", default=False, help="whether or not to use emdqn")
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
    emdqn = args.emdqn
    print(emdqn)
    print(args.double_q)
    print(args.dueling)
    print(args.prioritized)
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
        env = SimpleMonitor(env)
    else:
        env, monitored_env = make_env(args.env)

    subdir = (datetime.datetime.now()).strftime("%m-%d-%Y-%H:%M:%S") + " " + args.comment
    tf_writer = tf.summary.FileWriter(os.path.join(args.log_dir, subdir), tf.get_default_graph())
    value_summary = tf.Summary()
    qec_summary = tf.Summary()
    value_summary.value.add(tag='discount_reward_mean')
    # value_summary.value.add(tag='non_discount_reward_mean')
    # value_summary.value.add(tag='episode')

    qec_summary.value.add(tag='qec_mean')
    qec_summary.value.add(tag='qec_fount')
    # value_summary.value.add(tag='steps')
    # value_summary.value.add(tag='episodes')
    if args.seed > 0:
        set_global_seeds(args.seed)
        env.unwrapped.seed(args.seed)

    with U.make_session(4) as sess:
        # EMDQN
        if emdqn:
            ec_buffer = []
            buffer_size = 5000000
            latent_dim = 4
            input_dim = 220 * 220
            for i in range(env.action_space.n):
                ec_buffer.append(LRU_KNN(buffer_size, latent_dim, 'game'))
            rng = np.random.RandomState(123456)  # deterministic, erase 123456 for stochastic
            rp = rng.normal(loc=0, scale=1. / np.sqrt(latent_dim), size=(latent_dim, input_dim))
            qecwatch = []
            update_counter = 0
            qec_found = 0
            sequence = []
        tfout = open('result_%s_%s' % (args.env, str(emdqn)), 'w+')


        def update_kdtree():
            for a in range(env.action_space.n):
                ec_buffer[a].update_kdtree()


        def update_ec(sequence):
            Rtd = 0.
            for seq in reversed(sequence):
                s, a, r = seq
                z = np.dot(rp, s.flatten())
                Rtd = r + 0.99 * Rtd
                z = z.reshape((latent_dim))
                qd = ec_buffer[a].peek(z, Rtd, True)
                if qd == None:  # new action
                    ec_buffer[a].add(z, Rtd)


        # Create training graph and replay buffer
        act, train, update_target, debug, get_q_t_selected = deepq.build_train(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            q_func=dueling_model if args.dueling else model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
            gamma=0.99,
            grad_norm_clipping=10,
            double_q=args.double_q,
            emdqn=args.emdqn
        )


        def test_agent():  # TODO
            tenv, tenv_monitor = make_env(args.env)
            tenv.unwrapped.seed(args.seed)
            scores = []
            for i in range(30):
                tobs = tenv.reset()
                while True:
                    action = act(np.array(tobs)[None], stochastic=0.05)[0]
                    tobs, rew, done, info = tenv.step(action)
                    print(info)
                    if done and len(info["rewards"]) > 0:
                        score = info["rewards"][-1]
                        print("episode #%d: %.2f" % (i + 1, score))
                        scores.append(score)
                        tobs = tenv.reset()
                        break
            avg_score = np.mean(scores)
            print("avgscore: %.2f" % avg_score)
            return avg_score


        approximate_num_iters = args.num_steps / 4
        exploration = PiecewiseSchedule([
            (0, 1.0),
            (approximate_num_iters / 50, 0.1),
            (approximate_num_iters / 5, 0.01)
        ], outside_value=0.01)

        if args.prioritized:
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(approximate_num_iters, initial_p=args.prioritized_beta0, final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(args.replay_buffer_size,frame_history_len=1)

        U.initialize()
        update_target()
        num_iters = 0

        # Load the model
        state = maybe_load_model(savedir, container)
        if state is not None:
            num_iters, replay_buffer = state["num_iters"], state["replay_buffer"],
            monitored_env.set_state(state["monitor_state"])

        start_time, start_steps = None, None
        steps_per_iter = RunningAvg(0.999)
        iteration_time_est = RunningAvg(0.999)
        obs = env.reset()

        # Main trianing loop
        while True:
            num_iters += 1
            # Take action and store transition in the replay buffer.
            action= act(np.array(obs)[None], update_eps=exploration.value(num_iters))[0]
            new_obs, rew, done, info = env.step(action)
            if emdqn:
                sequence.append([np.array(obs), action, np.clip(rew, -1, 1), ])
            if args.prioritized:
                replay_buffer.add(obs, action, rew, new_obs, float(done))
            else:
                replay_buffer.add_batch([obs], [action], [rew], [float(done)])
            obs = new_obs
            if done:
                # EMDQN
                if emdqn:
                    update_ec(sequence)
                    sequence = []
                obs = env.reset()

            if (num_iters > max(5 * args.batch_size, args.replay_buffer_size // 20) and
                    num_iters % args.learning_freq == 0):
                # Sample a bunch of transitions from replay buffer
                if args.prioritized:
                    experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones, _ = replay_buffer.sample(args.batch_size)
                    weights = np.ones_like(rewards)
                # EMDQN
                if emdqn:
                    update_counter += 1
                    qec_input = get_q_t_selected(obses_t, actions)
                    for i in range(args.batch_size):
                        z = np.dot(rp, obses_t[i].flatten())
                        q = ec_buffer[actions[i]].peek(z, None, modify=False)
                        if q != None:
                            qec_input[i] = q
                            qecwatch.append(q)
                            qec_found += 1
                    qec_input = np.array(qec_input).reshape((args.batch_size))
                    if update_counter % 2000 == 1999:
                        print("qec_mean:", np.mean(qecwatch))
                        print("qec_fount: %.2f" % (1.0 * qec_found / args.batch_size / update_counter))
                        qec_summary.value[0].simple_value = np.mean(qecwatch)
                        qec_summary.value[1].simple_value = 1.0 * qec_found / args.batch_size / update_counter
                        qecwatch = []
                        tf_writer.add_summary(qec_summary, global_step=num_iters)


                # Minimize the error in Bellman's equation and compute TD-error
                if emdqn:
                    td_errors, qec_errors,summary = train(obses_t, actions, rewards, obses_tp1, dones, weights, qec_input)
                else:
                    td_errors,summary = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                tf_writer.add_summary(summary, global_step=num_iters)
                # Update the priorities in the replay buffer
                if args.prioritized:
                    new_priorities = np.abs(td_errors) + args.prioritized_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)
            # Update target network.
            if num_iters % args.target_update_freq == 0:  # NOTE: why not 10000?
                update_target()
                if emdqn:
                    update_kdtree()

            if start_time is not None:
                steps_per_iter.update(info['steps'] - start_steps)
                iteration_time_est.update(time.time() - start_time)
            start_time, start_steps = time.time(), info["steps"]

            # Save the model and training state.
            '''
            if num_iters > 0 and (num_iters % args.save_freq == 0 or info["steps"] > args.num_steps):
                maybe_save_model(savedir, container, {
                    'replay_buffer': replay_buffer,
                    'num_iters': num_iters,
                    'monitor_state': monitored_env.get_state()
                })
            '''

            if info["steps"] > args.num_steps:
                break

            if done:
                steps_left = args.num_steps - info["steps"]
                completion = np.round(info["steps"] / args.num_steps, 2)

                logger.record_tabular("% completion", completion)
                logger.record_tabular("steps", info["steps"])
                logger.record_tabular("iters", num_iters)
                logger.record_tabular("episodes", len(info["rewards"]))
                logger.record_tabular("reward (100 epi mean)", np.mean(info["rewards"][-100:]))
                if len(info["rewards"]) > 1:
                    tfout.write("%d, %.2f\n" % (info["steps"], np.mean(info["rewards"][-100:])))
                    tfout.flush()
                value_summary.value[0].simple_value = np.mean(info["rewards"][-100:])

                logger.record_tabular("exploration", exploration.value(num_iters))
                if args.prioritized:
                    logger.record_tabular("max priority", replay_buffer._max_priority)
                fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                                if steps_per_iter._value is not None else "calculating...")
                logger.dump_tabular()
                logger.log()
                logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                logger.log()
                tf_writer.add_summary(value_summary, global_step=num_iters)

            # if num_iters % 1000000 == 999999:
            #    avg_score = test_agent()
            #    tfout.write("%.2f\n" % avg_score)
            #    tfout.flush()
