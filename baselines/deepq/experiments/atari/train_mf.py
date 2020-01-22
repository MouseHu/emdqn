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
# when updating this to non-deperecated ones, it is important to
# copy over LazyFrames
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from baselines.common.azure_utils import Container
from baselines.deepq.experiments.atari.model import model, dueling_model
from baselines.deepq.experiments.atari.lru_knn import LRU_KNN


def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default="Pong", help="name of the game")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(1e7),
                        help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--knn", type=int, default=4, help="number of k nearest neighbours")
    parser.add_argument("--learning-freq", type=int, default=4,
                        help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=40000,
                        help="number of iterations between every target network update")
    # Bells and whistles
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default=None,
                        help="directory in which training state and model should be saved.")
    parser.add_argument("--save-azure-container", type=str, default=None,
                        help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
    parser.add_argument("--save-freq", type=int, default=1e6,
                        help="save model once every time this many iterations are completed")
    boolean_flag(parser, "load-on-start", default=True,
                 help="if true and model was previously saved then training will be resumed")
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
    env, monitored_env = make_env(args.env)
    eps = 1
    if args.seed > 0:
        set_global_seeds(args.seed)
        env.unwrapped.seed(args.seed)

    with U.make_session(4) as sess:
        # EMDQN

        ec_buffer = []
        buffer_size = 5000000
        latent_dim = 8
        input_dim = 84 * 84 * 4
        for i in range(env.action_space.n):
            ec_buffer.append(LRU_KNN(buffer_size, latent_dim, 'game'))
        rng = np.random.RandomState(123456)  # deterministic, erase 123456 for stochastic
        rp = rng.normal(loc=0, scale=1. / np.sqrt(latent_dim), size=(latent_dim, input_dim))
        qecwatch = []
        update_counter = 0
        qec_found = 0
        sequence = []
        tfout = open('result_%s_%s' % (args.env, "mfec"), 'w+')


        def act(ob, stochastic=0, update_eps=-1):
            global eps
            if update_eps >= 0:
                eps = update_eps
            if np.random.random() < max(stochastic,eps):
                action = np.random.randint(0, env.action_space.n)
                #print(eps,env.action_space.n,action)
                return action
            else:
                #print(eps,stochastic,np.random.rand(0, 1))
                q = []
                for a in range(env.action_space.n):
                    z = np.dot(rp, ob.flatten())
                    q.append(ec_buffer[a].knn_value(z, args.knn))
                #print("ec",eps,np.argmax(q),q)
                return np.argmax(q)


        def update_kdtree():
            for a in range(env.action_space.n):
                ec_buffer[a].update_kdtree()


        def update_ec(sequence):
            Rtd = 0.
            actions = [seq[1] for seq in sequence]
            print(np.bincount(actions))
            for seq in reversed(sequence):
                s, a, r = seq
                z = np.dot(rp, s.flatten())
                Rtd = r + 0.99 * Rtd
                z = z.reshape((latent_dim))
                qd = ec_buffer[a].peek(z, Rtd, True)
                if qd == None:  # new action
                    ec_buffer[a].add(z, Rtd)
            #for a in range(env.action_space.n):
            #    print(env.action_space.n,ec_buffer[a].curr_capacity)



        # Create training graph and replay buffer
        # act, train, update_target, debug, get_q_t_selected = deepq.build_train(
        #     make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
        #     q_func=dueling_model if args.dueling else model,
        #     num_actions=env.action_space.n,
        #     optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
        #     gamma=0.99,
        #     grad_norm_clipping=10,
        #     double_q=args.double_q,
        #     emdqn=args.emdqn
        # )

        def test_agent():  # TODO
            tenv, tenv_monitor = make_env(args.env)
            tenv.unwrapped.seed(args.seed)
            scores = []
            for i in range(30):
                tobs = tenv.reset()
                while True:
                    action = act(np.array(tobs)[None], stochastic=0.05)
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


        approximate_num_iters = args.num_steps /4
        exploration = PiecewiseSchedule([
            (0, 1.0),
            (approximate_num_iters / 10, 0.1),
            (approximate_num_iters / 5, 0.01)
        ], outside_value=0.01)

        U.initialize()
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
            action = act(np.array(obs)[None], update_eps=exploration.value(num_iters))
            new_obs, rew, done, info = env.step(action)
            sequence.append([np.array(obs), action, np.clip(rew, -1, 1), ])

            #replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs
            if done:
                # EMDQN
                update_ec(sequence)
                sequence = []
                obs = env.reset()

            # Update target network.
            if num_iters % args.target_update_freq == 0:  # NOTE: why not 10000?
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
                logger.record_tabular("exploration", exploration.value(num_iters))
                fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                                if steps_per_iter._value is not None else "calculating...")
                logger.dump_tabular()
                logger.log()
                logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                logger.log()

            # if num_iters % 1000000 == 999999:
            #    avg_score = test_agent()
            #    tfout.write("%.2f\n" % avg_score)
            #    tfout.flush()
