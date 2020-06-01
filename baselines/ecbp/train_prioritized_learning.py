from baselines.ecbp.util import *
from baselines.ecbp.agents.ecbp_agent import ECBPAgent
from baselines.ecbp.agents.ps_agent import PSAgent
from baselines.ecbp.agents.ps_mp_agent import PSMPAgent
from baselines.ecbp.agents.ps_mp_learning_agent import PSMPLearnAgent
from baselines.ecbp.agents.psmp_learning_target_agent import PSMPLearnTargetAgent
from baselines.ecbp.agents.kbps_mp_agent import KBPSMPAgent
from baselines.ecbp.agents.ec_debug_agent import ECDebugAgent
from baselines.ecbp.agents.kbps_agent import KBPSAgent
from baselines.ecbp.agents.ec_agent import ECAgent
from baselines.ecbp.agents.human_agent import HumanAgent
from baselines.ecbp.agents.hybrid_agent import HybridAgent, HybridAgent2
from baselines.ecbp.agents.graph.build_graph_contrast_target import *
from pyvirtualdisplay import Display
import sys
import logging

sys.setrecursionlimit(30000)
# from gym.wrappers.monitoring.video_recorder import VideoRecorder

if __name__ == '__main__':

    args = parse_args()
    if args.render:
        display = Display(visible=1, size=(640, 480))
        display.start()
    env = create_env(args)
    # env = GIFRecorder(video_path=args.video_path + "/{}/".format(args.comment), record_video=True, env=env)
    print("obs shape", env.observation_space.shape)
    subdir = (datetime.datetime.now()).strftime("%m-%d-%Y-%H:%M:%S") + " " + args.comment
    tf_writer = tf.summary.FileWriter(os.path.join(args.base_log_dir, args.log_dir, subdir), tf.get_default_graph())
    make_logger("ecbp", os.path.join(args.base_log_dir, args.log_dir, subdir, "logger.log"))
    make_logger("ec", os.path.join(args.base_log_dir, args.log_dir, subdir, "ec_logger.log"))
    exploration = PiecewiseSchedule([
        (0, 1),
        (args.end_training, 1.0),
        # (args.end_training+1, 1.0),
        # (args.end_training+1, 0.005),
        # (args.end_training + 10000, 1.0),
        (args.end_training + 500000, 0.05),
        (args.end_training + 1000000, 0.1),
        # (approximate_num_iters / 5, 0.1),
        # (approximate_num_iters / 3, 0.01)
    ], outside_value=0.1)
    try:
        num_actions = env.action_space.n
    except AttributeError:
        num_actions = env.unwrapped.pseudo_action_space.n
    obs_shape = env.observation_space.shape
    if obs_shape is None or obs_shape == (None,):
        obs_shape = env.unwrapped.observation_space.shape
    if type(obs_shape) is int:
        obs_shape = (obs_shape,)

    # ec_agent = ECAgent(rp_model if args.rp else contrastive_model, exploration, env.observation_space.shape,
    #                    args.lr,
    #                    args.buffer_size, env.action_space.n, args.latent_dim, args.gamma, args.knn, tf_writer)
    # agent = ec_agent
    # ps_agent = ECDebugAgent(rp_model if args.rp else contrastive_model,
    # ps_agent = PSMPLearnTargetAgent(representation_model_mlp if args.vector_input else representation_model_cnn,
    ps_agent = PSMPLearnTargetAgent(rp_model if args.rp else contrastive_model ,
                                    exploration,
                                    obs_shape, args.vector_input,
                                    args.lr,
                                    args.buffer_size, num_actions, args.latent_dim, args.gamma, args.knn,
                                    args.eval_epsilon, args.queue_threshold, args.batch_size,
                                    tf_writer)
    # human_agent = HumanAgent(
    #     {"w": 3, "s": 4, "d": 1, "a": 0, "x": 2, "p": 5, "3": 3, "4": 4, "1": 1, "0": 0, "2": 2, "5": 5})
    # agent = HybridAgent2(ps_agent, human_agent, 30)
    agent = ps_agent
    value_summary = tf.Summary()
    value_summary.value.add(tag='discount_reward_mean')
    value_summary.value.add(tag='non_discount_reward_mean')
    value_summary.value.add(tag='steps')
    value_summary.value.add(tag='episodes')

    with U.make_session(4) as sess:
        # EMDQN

        U.initialize()
        num_iters, eval_iters, num_episodes = 0, 0, 0
        non_discount_return, discount_return = [0.0], [0.0]
        # Load the model
        start_time, start_steps = time.time(), 0
        eval_start_steps = 0
        steps_per_iter, iteration_time_est = RunningAvg(0.999, 1), RunningAvg(0.999, 1)
        obs = env.reset()
        print_flag = True
        # Main training loop
        train_time, act_time, env_time, update_time, cur_time = 0, 0, 0, 0, time.time()
        while True:
            eval = (num_episodes % 10 == 9)
            if not eval:
                num_iters += 1
            else:
                eval_iters += 1
            # Take action and store transition in the replay buffer.
            action = agent.act(np.array(obs)[None], is_train=not eval)
            act_time += time.time() - cur_time
            cur_time = time.time()

            new_obs, rew, done, info = env.step(action)
            if args.render:
                env.render()
            env_time += time.time() - cur_time
            cur_time = time.time()

            agent.observe(action, rew, new_obs, done, train=not eval)
            update_time += time.time() - cur_time
            cur_time = time.time()

            if eval:
                non_discount_return[-1] += rew
                discount_return[-1] += rew * args.gamma ** (eval_iters - eval_start_steps)
            obs = new_obs
            if done:
                num_episodes += 1
                obs = env.reset()
                if eval:
                    non_discount_return.append(0.0)
                    discount_return.append(0.0)

            train_time += time.time() - cur_time
            cur_time = time.time()

            if start_time is not None:
                steps_per_iter.update(1)
                iteration_time_est.update(time.time() - start_time)
            start_time = time.time()

            if num_iters > args.num_steps:
                agent.finish()
                break

            if done:
                return_len = min(len(non_discount_return) - 1, 1)
                steps_left = args.num_steps - num_iters
                completion = np.round(num_iters / args.num_steps, 2)

                logger.record_tabular("% completion", completion)
                logger.record_tabular("iters", num_iters)
                logger.record_tabular("reward", np.mean(non_discount_return[-return_len - 1:-1]))
                logger.record_tabular("discount reward", np.mean(discount_return[-return_len - 1:-1]))
                logger.record_tabular("num episode", num_episodes)
                logger.record_tabular("update time", update_time)
                logger.record_tabular("train time", train_time)
                logger.record_tabular("act_time", act_time)
                logger.record_tabular("env_time", env_time)
                logger.record_tabular("eval", num_episodes % 2 == 0)

                if eval:
                    total_steps = num_iters - args.end_training
                    value_summary.value[0].simple_value = np.mean(discount_return[-return_len - 1:-1])
                    value_summary.value[1].simple_value = np.mean(non_discount_return[-return_len - 1:-1])
                    value_summary.value[2].simple_value = num_iters
                    value_summary.value[3].simple_value = num_episodes
                    tf_writer.add_summary(value_summary, global_step=total_steps)

                logger.record_tabular("exploration", exploration.value(num_iters))
                fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                                if steps_per_iter._value is not None else 1 / (float(iteration_time_est) + 1e-6))
                logger.dump_tabular()
                logger.log()
                logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                logger.log()

                eval = (num_episodes % 10 == 9)
                if not eval:
                    start_steps = num_iters
                else:
                    eval_start_steps = eval_iters
                total_steps = num_iters - args.end_training

            # tf_writer.add_summary(qec_summary, global_step=total_steps)
            cur_time = time.time()
