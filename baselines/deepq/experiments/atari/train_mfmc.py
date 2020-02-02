import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import tempfile
import time
import cv2
import sys

cwd = os.getcwd()
cwd = '/'.join(cwd.split('/')[:-4])
temp = sys.path
temp.append('')
temp[1:] = temp[0:-1]
temp[0] = cwd
print(sys.path)

import baselines.common.tf_util as U
import datetime
from baselines import logger
from baselines import deepq
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
from baselines.deepq.experiments.atari.model import contrastive_model
from baselines.deepq.experiments.atari.lru_knn_mc import LRU_KNN_MC


def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default="Pong", help="name of the game")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--momentum", type=float, default=0.999, help="momentum for momentum contrastive encoder")
    parser.add_argument("--negative-samples", type=int, default=10, help="numbers for negative samples")
    parser.add_argument("--knn", type=int, default=4, help="number of k nearest neighbours")
    parser.add_argument("--num-steps", type=int, default=int(1e7),
                        help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="number of stransitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=4,
                        help="number of iterations between every optimization step")
    parser.add_argument("--encoder-update-freq", type=int, default=1000,
                        help="number of iterations between every target network update")
    parser.add_argument("--tree-update-freq", type=int, default=10000,
                        help="number of iterations between every target network update")
    # Bells and whistles
    boolean_flag(parser, "prioritized", default=False, help="whether or not to use prioritized replay buffer")
    parser.add_argument("--prioritized-alpha", type=float, default=0.6,
                        help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized-beta0", type=float, default=0.4,
                        help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps", type=float, default=1e-6,
                        help="eps parameter for prioritized replay buffer")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default=None,
                        help="directory in which training state and model should be saved.")
    parser.add_argument("--save-azure-container", type=str, default=None,
                        help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
    parser.add_argument("--save-freq", type=int, default=1e6,
                        help="save model once every time this many iterations are completed")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="latent_dim")
    parser.add_argument("--comment", type=str, default=datetime.datetime.now().strftime("%I-%M_%B-%d-%Y"),
                        help="discription for this experiment")
    parser.add_argument("--log_dir", type=str, default="./tflogs",
                        help="directory in which training state and model should be saved.")
    boolean_flag(parser, "load-on-start", default=True,
                 help="if true and model was previously saved then training will be resumed")

    # EMDQN

    boolean_flag(parser, "predict", default=False, help="whether or not to use prediction")
    boolean_flag(parser, "learning", default=False, help="whether or not to learn encoder")

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


def switch_first_half(obs, obs_next, batch_size):
    half_size = int(batch_size / 2)
    tmp = obs[:half_size, ...]
    obs[:half_size, ...] = obs_next[:half_size, ...]
    obs_next[:half_size, ...] = tmp
    return obs, obs_next


if __name__ == '__main__':

    args = parse_args()
    print("predict value:{} learning:{}".format(args.predict, args.learning))
    tf.random.set_random_seed(args.seed)
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
    if args.seed > 0:
        set_global_seeds(args.seed)
        env.unwrapped.seed(args.seed)

    subdir = (datetime.datetime.now()).strftime("%m-%d-%Y-%H:%M:%S") + " " + args.comment
    tf_writer = tf.summary.FileWriter(os.path.join(args.log_dir, subdir), tf.get_default_graph())
    value_summary = tf.Summary()
    qec_summary = tf.Summary()
    value_summary.value.add(tag='reward_mean')
    qec_summary.value.add(tag='qec_mean')
    qec_summary.value.add(tag='qec_fount')
    value_summary.value.add(tag='steps')

    with U.make_session(4) as sess:
        # EMDQN
        ec_buffer = []
        buffer_size = 1000000
        latent_dim = args.latent_dim
        input_dim = 84 * 84 * 4
        # rng = np.random.RandomState(123456)  # deterministic, erase 123456 for stochastic
        # rp = rng.normal(loc=0, scale=1. / np.sqrt(latent_dim), size=(latent_dim, input_dim))
        for i in range(env.action_space.n):
            ec_buffer.append(LRU_KNN_MC(buffer_size, latent_dim, latent_dim, 'game'))
        # rng = np.random.RandomState(123456)  # deterministic, erase 123456 for stochastic
        # rp = rng.normal(loc=0, scale=1. / np.sqrt(latent_dim), size=(latent_dim, input_dim))
        qec_watch = []
        update_counter = 0
        qec_found = 0
        sequence = []
        tfout = open(
            './results/result_%s_mfmc_predict%s_%s' % (args.env, str(args.predict), args.comment), 'w+')
        visualout = './visual/'


        def act(ob, stochastic=0, update_eps=-1):
            global eps,qec_found,qec_watch
            z = z_func(np.array(ob))
            h = hash_func(np.array(ob))
            # print(z[0].shape,h[0].shape)
            if update_eps >= 0:
                eps = update_eps
            if np.random.random() < max(stochastic, eps):
                action = np.random.randint(0, env.action_space.n)
                # print(eps,env.action_space.n,action)
                return action, z, h
            else:
                # print(eps,stochastic,np.random.rand(0, 1))
                q = []
                for a in range(env.action_space.n):
                    #print(z[0].shape,h[0].shape)
                    q_value, found = ec_buffer[a].act_value(z[0][0], h[0][0], args.knn)
                    q.append(q_value)
                    if found:
                        qec_found += 1
                        qec_watch.append(q_value)
                # print("ec",eps,np.argmax(q),q)
                return np.argmax(q), z, h


        def update_kdtree():
            for a in range(env.action_space.n):
                ec_buffer[a].update_kdtree()


        def update_ec(sequence):
            obses, acts, rs = list(zip(*sequence))
            Rtds = []
            Rtd = 0
            for r in reversed(rs):
                Rtd = r + 0.99 * Rtd
                Rtds.append(Rtd)
            Rtds = Rtds[::-1]
            obses = np.array([np.array(ob) for ob in obses])
            hashes = hash_func(obses)
            zs = encoder_z_func(obses)
            # print(obses.shape,len(hashes[0]),len(zs[0]),len(Rtds))
            for a, z, h, Rtd in zip(acts, zs[0], hashes[0], Rtds):
                qd = ec_buffer[a].peek(h, Rtd, True)
                if qd == None:  # new action
                    # print("add",z,h)
                    ec_buffer[a].add(z, h, Rtd)


        # Create training graph and replay buffer
        hash_func, z_func, encoder_z_func, update_encoder, train = deepq.build_train_mfmc(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            model_func=contrastive_model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
            gamma=0.99,
            grad_norm_clipping=10,
            input_dim=84 * 84 * 4,
            K=args.negative_samples,
            predict=args.predict
        )


        def test_agent():  # TODO
            tenv, tenv_monitor = make_env(args.env)
            tenv.unwrapped.seed(args.seed)
            scores = []
            for i in range(30):
                tobs = tenv.reset()
                while True:
                    action, z_test, h_test = act(np.array(tobs)[None], stochastic=0.05)
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
            replay_buffer = ReplayBufferHash(args.replay_buffer_size)

        U.initialize()
        update_encoder([0])
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
            action, z, h = act(np.array(obs)[None], update_eps=exploration.value(num_iters))
            new_obs, rew, done, info = env.step(action)
            new_h = hash_func(np.array(new_obs)[None, :])
            # EMDQN

            sequence.append([obs, action, np.clip(rew, -1, 1)])
            if args.learning:
                replay_buffer.add(obs, h, action, rew, new_obs, new_h, float(done))
            obs = new_obs
            if done:
                # EMDQN
                update_ec(sequence)
                sequence = []
                obs = env.reset()

            if (num_iters > max(5 * args.batch_size, args.replay_buffer_size // 20) and
                num_iters % args.learning_freq == 0) and args.learning:
                # Sample a bunch of transitions from replay buffer
                # if args.prioritized:
                #     experience_contra = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters))
                #     (obses_contra, actions, rewards_contra, obses_contra, dones_contra, weights_contra,
                #      batch_idxes_contra) = experience_contra
                #     obses_anchor, obses_pos = switch_first_half(obses_contra, obses_contra_tp1, args.batch_size)
                #     if args.predict:
                #         experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters))
                #         (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                #
                # else:
                if args.predict:
                    obses_t, hashes_t, actions, rewards, obses_tp1, hashes_tp1, dones = replay_buffer.sample(
                        args.batch_size)
                obses_contra, hashes_contra, actions_contra, rewards_contra, obses_contra_tp1, hashes_contra_tp1, dones_contra = replay_buffer.sample(
                    args.batch_size)
                # obses_anchor, obses_pos = switch_first_half(obses_contra, obses_contra_tp1, args.batch_size)
                # hashes_anchor, hashes_pos = switch_first_half(hashes_contra, hashes_contra_tp1, args.batch_size)
                obses_anchor, obses_pos = obses_contra, obses_contra_tp1
                hashes_anchor, hashes_pos = hashes_contra, hashes_contra_tp1
                # EMDQN
                neg_keys = [
                    ec_buffer[actions_contra[i]].sample_keys([hashes_anchor[i], hashes_pos[i]], args.negative_samples)
                    for i in range(args.batch_size)]
                update_counter += 1
                if args.predict:
                    value_input = np.zeros(args.batch_size)
                    hs = hash_func(obses_t)
                    value_found = [ec_buffer[actions[i]].act_value(z[i], hs[i][0], args.knn) for i in
                                   range(args.batch_size)]
                    values, founds = list(zip(*value_found))
                    qec_found += sum(founds)
                    qec_watch += sum(values[founds])
                if update_counter % 200 == 199:
                    print("qec_mean:", np.mean(qec_watch))
                    print("qec_fount: %.2f" % (1.0 * qec_found / args.batch_size / update_counter))

                    qec_summary.value[0].simple_value = np.mean(qec_watch)
                    qec_summary.value[1].simple_value = 1.0 * qec_found / args.batch_size / update_counter
                    tf_writer.add_summary(qec_summary, global_step=info["steps"])
                    qec_watch = []
                    qec_found = 0

                # Minimize the error in Bellman's equation and compute TD-error
                if not args.predict:
                    inputs = [[100], obses_anchor, obses_pos, neg_keys]
                else:
                    inputs = [[100], obses_anchor, obses_pos, neg_keys, obses_t, value_input]

                total_errors, summary = train(*inputs)

                # Update the priorities in the replay buffer
                # if args.prioritized:
                #     new_priorities = np.abs(total_errors) + args.prioritized_eps
                #     replay_buffer.update_priorities(batch_idxes, new_priorities)

                tf_writer.add_summary(summary, global_step=info["steps"])

                # tf_writer.add_summary(summary,global_step=info["steps"])
                # Update target network.
                # if num_iters % args.target_update_freq == 0:  # NOTE: why not 10000?
            if num_iters % args.tree_update_freq == 0:
                update_kdtree()
            if num_iters % args.encoder_update_freq == 0:
                update_encoder([args.momentum])
            if num_iters == 100:
                print("saving")
                obses, acts, rs = list(zip(*sequence))
                for i, obs in enumerate(obses):
                    cv2.imwrite("./visual/{}.png".format(i), np.array(obs)[:, :, 0])

            # sample input to visualize

            if start_time is not None:
                steps_per_iter.update(info['steps'] - start_steps)
                iteration_time_est.update(time.time() - start_time)
            start_time, start_steps = time.time(), info["steps"]
            value_summary.value[1].simple_value = num_iters

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
                value_summary.value[0].simple_value = np.mean(info["rewards"][-100:])
                if len(info["rewards"]) > 1:
                    np.mean(info["rewards"][-100:])
                    tfout.write("%d, %.2f\n" % (info["steps"], np.mean(info["rewards"][-100:])))
                    tfout.flush()
                logger.record_tabular("exploration", exploration.value(num_iters))
                if args.prioritized:
                    logger.record_tabular("max priority", replay_buffer._max_priority)
                fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                                if steps_per_iter._value is not None else "calculating...")
                logger.dump_tabular()
                logger.log()
                logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                logger.log()
            tf_writer.add_summary(value_summary, global_step=info["steps"])

            # if num_iters % 1000000 == 1:
            #     avg_score = test_agent()
            #     tfout.write("test:%.2f\n" % avg_score)
            #     tfout.flush()
