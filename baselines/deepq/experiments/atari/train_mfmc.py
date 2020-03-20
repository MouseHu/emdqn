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

from baselines.deepq.dqn_utils import *
import baselines.common.tf_util as U
import datetime
from baselines import logger
from baselines import deepq
from baselines.common.atari_lib import create_atari_environment
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
from baselines.deepq.experiments.atari.model import contrastive_model, contrastive_model_general
from baselines.deepq.experiments.atari.lru_knn_combine import LRU_KNN_COMBINE


def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default="Pong", help="name of the game")
    parser.add_argument("--seed", type=int, default=int(time.time()), help="which seed to use")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--momentum", type=float, default=0.999, help="momentum for momentum contrastive encoder")
    parser.add_argument("--negative-samples", type=int, default=1, help="numbers for negative samples")
    parser.add_argument("--knn", type=int, default=4, help="number of k nearest neighbours")
    parser.add_argument("--gamma", type=int, default=0.99, help="which seed to use")
    parser.add_argument("--num-steps", type=int, default=int(5e6),
                        help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=64,
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
    parser.add_argument("--prioritized-beta", type=float, default=0.4,
                        help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps", type=float, default=1e-6,
                        help="eps parameter for prioritized replay buffer")
    # Checkpointing
    parser.add_argument('--map_config', type=str,
                        help='The map and config you want to run in MonsterKong.',
                        default='../../../ple/configs/config_ppo_mk_large.py')
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
    boolean_flag(parser, "learning", default=True, help="whether or not to learn encoder")

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
    env = SimpleMonitor(env)
    subdir = (datetime.datetime.now()).strftime("%m-%d-%Y-%H:%M:%S") + " " + args.comment
    tf_writer = tf.summary.FileWriter(os.path.join(args.log_dir, subdir), tf.get_default_graph())
    value_summary = tf.Summary()
    qec_summary = tf.Summary()
    value_summary.value.add(tag='reward_mean')
    value_summary.value.add(tag='discount_reward_mean')
    value_summary.value.add(tag='non_discount_reward_mean')
    qec_summary.value.add(tag='qec_mean')
    qec_summary.value.add(tag='qec_fount')
    value_summary.value.add(tag='steps')

    with U.make_session(4) as sess:
        # EMDQN

        buffer_size = 50000
        latent_dim = args.latent_dim
        input_dim = 220 * 220 * 1 if args.env == "MK" else 84 * 84 * 4
        input_dims = (220, 220, 1) if args.env == "MK" else (84, 84, 4)
        # rng = np.random.RandomState(123456)  # deterministic, erase 123456 for stochastic
        # rp = rng.normal(loc=0, scale=1. / np.sqrt(latent_dim), size=(latent_dim, input_dim))
        ec_buffer = LRU_KNN_COMBINE(env.action_space.n, buffer_size, latent_dim, latent_dim, input_dims, 'game')
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
            global eps, qec_found, qec_watch
            z = z_func(np.array(ob))
            h = hash_func(np.array(ob))
            # print(z[0].shape,h[0].shape)
            if update_eps >= 0:
                eps = update_eps
            if np.random.random() < max(stochastic, eps):
                action = np.random.randint(0, env.action_space.n)
                # for a in range(env.action_space.n):
                #     print(np.array(z).shape)
                #     q, count = ec_buffer[a].peek(None,h[0][0], 0, modify=False)
                #     print("random q", q)
                #     if q is not None:
                #         qec_watch.append(q)
                #         qec_found += 1
                # print(eps,env.action_space.n,action)
                return action, z, h
            else:
                # print(eps,stochastic,np.random.rand(0, 1))
                q = []
                for a in range(env.action_space.n):
                    q_value = ec_buffer.knn_value(a, z[0][0], args.knn)
                    q.append(q_value)

                q_max = np.max(q)
                # print("optimistic q", optimistic_q.shape, np.where(optimistic_q == q_max))
                max_action = np.where(q == q_max)[0]
                # print(max_action)
                action_selected = np.random.randint(0, len(max_action))
                # print("ec",eps,np.argmax(q),q)
                return max_action[action_selected], z, h
                # return np.argmax(q), z, h


        def update_kdtree():
            ec_buffer.update_kdtree()


        def update_ec(sequence):
            obses, acts, zs, hs, rs = list(zip(*sequence))
            Rtds = []
            Rtd = 0
            for r in reversed(rs):
                Rtd = r + 0.99 * Rtd
                Rtds.append(Rtd)
            Rtds = Rtds[::-1]
            obses = np.array([np.array(ob) for ob in obses])
            # hashes = hash_func(obses)
            # zs = z_func(obses)
            # print(obses.shape,len(hashes[0]),len(zs[0]),len(Rtds))
            prev_id, prev_action = -1, -1
            for a, obs, z, h, Rtd in zip(acts, obses, zs, hs, Rtds):
                # z = z_func([obs])[0]
                # h = hash_func([obs])[0][0]
                # print(np.array(h).shape)
                qd, prev_id_tmp = ec_buffer.peek(a, z[0][0], h[0][0], Rtd, True, False, prev_id, prev_action)
                if qd == None:  # new action
                    # print("add",z,h)
                    prev_id = ec_buffer.add(a, z[0][0], h[0][0], Rtd, prev_id, prev_action, obs)
                else:
                    prev_id = prev_id_tmp
                prev_action = a


        # Create training graph and replay buffer
        hash_func, z_func, train = deepq.build_train_mfmc(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            model_func=contrastive_model_general,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
            # optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
            gamma=0.99,
            grad_norm_clipping=10,
            input_dim=input_dim,
            batch_size=args.batch_size,
            K=args.negative_samples,
            predict=args.predict
        )

        tf_writer.add_graph(sess.graph)

        approximate_num_iters = args.num_steps
        exploration = PiecewiseSchedule([
            (0, 1.0),
            (400000, 0.05),
            (800000, 0.01)
        ], outside_value=0.01)

        # if args.prioritized:
        #     replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
        #     beta_schedule = LinearSchedule(approximate_num_iters, initial_p=args.prioritized_beta0, final_p=1.0)
        # else:
        #     replay_buffer = ReplayBufferHash(args.replay_buffer_size)

        U.initialize()
        # update_encoder([0])
        num_iters = 0

        # Load the model
        state = maybe_load_model(savedir, container)
        # if state is not None:
        #     num_iters, replay_buffer = state["num_iters"], state["replay_buffer"],
        #     # monitored_env.set_state(state["monitor_state"])

        start_time, start_steps = time.time(), 0
        steps_per_iter = RunningAvg(0.999)
        iteration_time_est = RunningAvg(0.999)
        obs = env.reset()
        non_discount_return = [0.0]
        discount_return = [0.0]
        train_time = 0
        act_time = 0
        env_time = 0
        update_time = 0
        cur_time = time.time()
        update_lr = 1e-3
        # Main trianing loop
        while True:
            num_iters += 1
            # Take action and store transition in the replay buffer.
            action, z, h = act(np.array(obs)[None], update_eps=exploration.value(num_iters))
            act_time += time.time() - cur_time
            cur_time = time.time()
            new_obs, rew, done, info = env.step(action)
            env_time += time.time() - cur_time
            cur_time = time.time()
            new_h = hash_func(np.array(new_obs)[None, :])
            # EMDQN
            non_discount_return[-1] += rew
            discount_return[-1] += rew * args.gamma ** (num_iters - start_steps)
            sequence.append([obs, action, z, h, np.clip(rew, -1, 1)])
            # if args.learning:
            #     replay_buffer.add(obs, h, action, rew, new_obs, new_h, float(done))
            obs = new_obs
            if done:
                # EMDQN

                update_ec(sequence)
                update_time += time.time() - cur_time
                cur_time = time.time()
                sequence = []
                obs = env.reset()
                non_discount_return.append(0.0)
                discount_return.append(0.0)

            if (num_iters > 500 * args.batch_size) and (num_iters % args.learning_freq == 0) and args.learning:
                place_anchor, place_pos, place_neg, obs_anchor, key_anchor, key_pos, key_neg = ec_buffer.sample(
                    args.batch_size,
                    args.negative_samples)

                # print(np.array(key_neg).shape)
                # print("training")
                inputs = [[1], obs_anchor, key_pos, key_neg, key_anchor]
                total_loss, summary, z_anchor, z_pos, z_neg = train(*inputs)
                # update dictionary
                ec_buffer.update(place_anchor, z_anchor)
                z_pos = np.nan_to_num(z_pos, copy=False)
                z_neg = np.nan_to_num(z_neg, copy=False)
                z_pos = np.array(z_pos[0]) * update_lr + np.array(key_pos)
                z_neg = np.array(z_neg[0]) * update_lr + np.array(key_neg)

                ec_buffer.update(place_pos, z_pos)
                for j in range(args.batch_size):
                    ec_buffer.update(place_neg[j], z_neg[j])
                tf_writer.add_summary(summary, global_step=info["steps"])

                # tf_writer.add_summary(summary,global_step=info["steps"])
                # Update target network.
                # if num_iters % args.target_update_freq == 0:  # NOTE: why not 10000?
            train_time += time.time() - cur_time
            cur_time = time.time()
            if num_iters % args.tree_update_freq == 0:
                update_kdtree()

            if start_time is not None:
                steps_per_iter.update(1)
                iteration_time_est.update(time.time() - start_time)
            start_time = time.time()
            value_summary.value[3].simple_value = num_iters

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
                return_len = min(len(non_discount_return) - 1, 100)
                steps_left = args.num_steps - info["steps"]
                completion = np.round(info["steps"] / args.num_steps, 2)

                logger.record_tabular("% completion", completion)
                logger.record_tabular("steps", info["steps"])
                logger.record_tabular("iters", num_iters)
                logger.record_tabular("episodes", len(info["rewards"]))
                logger.record_tabular("qec_mean", np.mean(qec_watch))
                logger.record_tabular("qec_proportion", qec_found / (num_iters - start_steps + 1))
                logger.record_tabular("reward (100 epi mean)", np.mean(info["rewards"][-100:]))
                logger.record_tabular("update time", update_time)
                logger.record_tabular("train time", train_time)
                logger.record_tabular("act_time", act_time)
                logger.record_tabular("env_time", env_time)
                # value_summary.value[0].simple_value = np.mean(info["rewards"][-100:])
                value_summary.value[1].simple_value = np.mean(discount_return[-return_len - 1:-1])
                value_summary.value[2].simple_value = np.mean(non_discount_return[-return_len - 1:-1])
                value_summary.value[0].simple_value = np.mean(np.mean(info["rewards"][-100:]))
                qec_summary.value[0].simple_value = np.mean(qec_watch)
                qec_summary.value[1].simple_value = qec_found / (num_iters - start_steps + 1)
                # if len(info["rewards"]) > 1:
                #                 #     np.mean(info["rewards"][-100:])
                #                 #     tfout.write("%d, %.2f\n" % (info["steps"], np.mean(info["rewards"][-100:])))
                #                 #     tfout.flush()
                logger.record_tabular("exploration", exploration.value(num_iters))
                # if args.prioritized:
                #     logger.record_tabular("max priority", replay_buffer._max_priority)
                fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                                if steps_per_iter._value is not None else "calculating...")
                logger.dump_tabular()
                logger.log()
                logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                logger.log()
                qec_watch = []
                qec_found = 0
                start_steps = num_iters
            tf_writer.add_summary(value_summary, global_step=info["steps"])
            cur_time = time.time()
            # if num_iters % 1000000 == 1:
            #     avg_score = test_agent()
            #     tfout.write("test:%.2f\n" % avg_score)
            #     tfout.flush()
