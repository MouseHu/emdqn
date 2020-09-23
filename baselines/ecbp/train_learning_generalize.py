from pyvirtualdisplay import Display

# display = Display(visible=1, size=(960, 640))
# display.start()
from baselines.ecbp.util import *
from baselines.ecbp.agents.ecbp_agent import ECBPAgent
from baselines.ecbp.agents.ps_agent import PSAgent
from baselines.ecbp.agents.ps_mp_agent import PSMPAgent
from baselines.ecbp.agents.ps_mp_learning_agent import PSMPLearnAgent
from baselines.ecbp.agents.psmp_learning_target_agent import PSMPLearnTargetAgent
from baselines.ecbp.agents.mer_attention_agent import MERAttentionAgent
from baselines.ecbp.agents.mer_bvae_attention_agent import BVAEAttentionAgent
from baselines.ecbp.agents.kbps_mp_agent import KBPSMPAgent
from baselines.ecbp.agents.ec_debug_agent import ECDebugAgent
from baselines.ecbp.agents.kbps_agent import KBPSAgent
from baselines.ecbp.agents.ec_agent import ECAgent
from baselines.ecbp.agents.ec_learning_agent import ECLearningAgent
from baselines.ecbp.agents.human_agent import HumanAgent
from baselines.ecbp.agents.hybrid_agent import HybridAgent, HybridAgent2
from baselines.ecbp.agents.graph.build_graph_contrast_target import *
from baselines.ecbp.test.buffer_test import buffertest
import sys
import logging

# from gym.wrappers.monitoring.video_recorder import VideoRecorder

if __name__ == '__main__':

    args = parse_args()
    # if args.render:
    vars(args).update({'number': 1})
    vars(args).update({'noise_size': 1})
    train_env = create_env(args)
    vars(args).update({'number': 2})
    # vars(args).update({'env_name': "large_2"})
    vars(args).update({'env_name': "fourrooms_noise"})
    vars(args).update({'noise_size': 10000})

    test_env = create_env(args)
    # env = GIFRecorder(video_path=args.video_path + "/{}/".format(args.comment), record_video=True, env=env)
    subdir = (datetime.datetime.now()).strftime("%m-%d-%Y-%H:%M:%S") + "_" + args.comment
    tfdir = os.path.join(args.base_log_dir, args.log_dir, subdir)
    agentdir = os.path.join(args.base_log_dir, args.agent_dir, subdir)
    tf_writer = tf.summary.FileWriter(tfdir, tf.get_default_graph())
    make_logger("ecbp", os.path.join(args.base_log_dir, args.log_dir, subdir, "logger.log"))
    make_logger("ec", os.path.join(args.base_log_dir, args.log_dir, subdir, "ec_logger.log"))
    # os.path.join(args.base_log_dir, args.log_dir, subdir, "logger.log")
    exploration = PiecewiseSchedule([
        (0, 1),
        (args.end_training, 1.0),
        # (args.end_training+1, 1.0),
        # (args.end_training+1, 0.005),
        # (args.end_training + 10000, 1.0),
        (args.end_training + 100000, 0.2),
        (args.end_training + 200000, 0.1),
        # (approximate_num_iters / 5, 0.1),
        # (approximate_num_iters / 3, 0.01)
    ], outside_value=0.1)
    try:
        num_actions = train_env.action_space.n
    except AttributeError:
        num_actions = train_env.unwrapped.pseudo_action_space.n

    obs_shape = train_env.observation_space.shape
    if obs_shape is None or obs_shape == (None,):
        obs_shape = train_env.unwrapped.observation_space.shape
    if obs_shape is None or obs_shape == (None,):
        obs_shape = train_env.unwrapped.observation_space['images'].shape
    if type(obs_shape) is int:
        obs_shape = (obs_shape,)

    # ec_agent = ECAgent(representation_model_mlp if args.vector_input else representation_model_cnn, exploration, env.observation_space.shape,
    #                    args.lr,
    #                    args.buffer_size, num_actions, args.latent_dim, args.gamma, args.knn, tf_writer)
    # agent = ec_agent
    # ps_agent = ECDebugAgent(rp_model if args.rp else contrastive_model,
    # ps_agent = PSMPLearnTargetAgent(
        # representation_model_mlp if args.vector_input else unit_representation_model_cnn,
    ps_agent = MERAttentionAgent(
        # ps_agent = ECLearningAgent(
        representation_model_mlp if args.vector_input else representation_with_mask_model_cnn,
        # bvae_encoder,bvae_decoder,
        # rp_model if args.rp else contrastive_model ,
        exploration,
        obs_shape, args.vector_input,
        args.lr,
        args.buffer_size, num_actions, args.latent_dim, args.gamma, args.knn,
        args.eval_epsilon, args.queue_threshold, args.batch_size, args.density, args.trainable, args.negative_samples,
        # debug=True, debug_dir=agentdir,
        tf_writer=tf_writer)
    # ps_agent = KBPSMPAgent(representation_model_mlp if args.vector_input else representation_model_cnn,
    #                                 # ps_agent = PSMPLearnTargetAgent(rp_model if args.rp else contrastive_model ,
    #                                 exploration,
    #                                 obs_shape, args.vector_input,
    #                                 args.lr,
    #                                 args.buffer_size, num_actions, args.latent_dim, args.gamma, args.knn,
    #                                 args.eval_epsilon, args.queue_threshold, args.batch_size,
    #                                 tf_writer)
    # human_agent = HumanAgent(
    #     {"w": 3, "s": 4, "d": 1, "a": 0, "x": 2, "p": 5, "3": 3, "4": 4, "1": 1, "0": 0, "2": 2, "5": 5})
    # agent = HybridAgent2(ps_agent, human_agent, 30)
    agent = ps_agent
    value_summary_train = tf.Summary()
    value_summary_train.value.add(tag='discount_reward_mean')
    value_summary_train.value.add(tag='non_discount_reward_mean')
    value_summary_train.value.add(tag='steps')
    value_summary_train.value.add(tag='episodes')
    value_summary_train.value.add(tag='discount_reward_mean_training')
    value_summary_train.value.add(tag='non_discount_reward_mean_training')

    value_summary_test = tf.Summary()
    value_summary_test.value.add(tag='discount_reward_mean_generalize')
    value_summary_test.value.add(tag='non_discount_reward_mean_generalize')
    value_summary_test.value.add(tag='steps_generalize')
    value_summary_test.value.add(tag='episodes_generalize')
    value_summary_test.value.add(tag='discount_reward_mean_generalize_training')
    value_summary_test.value.add(tag='non_discount_reward_mean_generalize_training')

    with U.make_session(4) as sess:
        # EMDQN

        U.initialize()
        saver = tf.train.Saver()


        def run(env, test=False):
            total_steps_required = args.num_steps if not test else args.test_num_steps
            if test:
                agent.empty_buffer()
                agent.trainable = False
            num_iters, eval_iters, num_episodes = 0, 0, 0
            non_discount_return, discount_return = [0.0], [0.0]
            non_discount_return_eval, discount_return_eval = [0.0], [0.0]
            # Load the model
            start_time, start_steps = time.time(), 0
            steps_per_iter, iteration_time_est = RunningAvg(0.999, 1), RunningAvg(0.999, 1)
            obs = env.reset()
            print("in main", obs)
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

                if num_iters % 10000 == 0:
                    # print(tf.global_variables())
                    agent.save(agentdir, sess, saver)
                if eval:
                    non_discount_return_eval[-1] += rew
                    discount_return_eval[-1] += rew * args.gamma ** (eval_iters - start_steps)
                else:
                    non_discount_return[-1] += rew
                    discount_return[-1] += rew * args.gamma ** (num_iters - start_steps)
                obs = new_obs
                if done:
                    num_episodes += 1
                    if eval:
                        non_discount_return_eval.append(0.0)
                        discount_return_eval.append(0.0)
                    else:
                        non_discount_return.append(0.0)
                        discount_return.append(0.0)
                    obs = env.reset()

                train_time += time.time() - cur_time

                if start_time is not None:
                    steps_per_iter.update(1)
                    iteration_time_est.update(time.time() - start_time)
                start_time = time.time()

                if num_iters > total_steps_required and done:
                    # buffertest(agent, args.comment)
                    if test:
                        agent.finish()
                    break

                if done:
                    return_len = min(len(non_discount_return) - 1, 1)
                    return_len_eval = min(len(non_discount_return_eval) - 1, 1)
                    steps_left = args.num_steps - num_iters
                    completion = np.round(num_iters / args.num_steps, 2)

                    logger.record_tabular("% completion", completion)
                    logger.record_tabular("iters", num_iters)
                    logger.record_tabular("return", np.mean(non_discount_return[-return_len - 1:-1]))
                    logger.record_tabular("return_eval", np.mean(non_discount_return_eval[-return_len_eval - 1:-1]))
                    logger.record_tabular("discount return", np.mean(discount_return[-return_len - 1:-1]))
                    logger.record_tabular("discount return_eval",
                                          np.mean(discount_return_eval[-return_len_eval - 1:-1]))
                    logger.record_tabular("num episode", num_episodes)
                    logger.record_tabular("update time", update_time)
                    logger.record_tabular("train time", train_time)
                    logger.record_tabular("act_time", act_time)
                    logger.record_tabular("env_time", env_time)
                    logger.record_tabular("eval", num_episodes % 2 == 0)

                    if test:
                        value_summary = value_summary_test
                    else:
                        value_summary = value_summary_train
                    if eval:
                        total_steps = num_iters - args.end_training

                        value_summary.value[0].simple_value = np.mean(discount_return_eval[-return_len_eval - 1:-1])
                        value_summary.value[1].simple_value = np.mean(
                            non_discount_return_eval[-return_len_eval - 1:-1])
                        value_summary.value[2].simple_value = num_iters
                        value_summary.value[3].simple_value = num_episodes
                        tf_writer.add_summary(value_summary, global_step=total_steps)
                    else:
                        total_steps = num_iters - args.end_training
                        value_summary.value[4].simple_value = np.mean(discount_return[-return_len - 1:-1])
                        value_summary.value[5].simple_value = np.mean(non_discount_return[-return_len - 1:-1])
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
                        start_steps = eval_iters
                    total_steps = num_iters - args.end_training

                # tf_writer.add_summary(qec_summary, global_step=total_steps)
                cur_time = time.time()


        run(train_env, False)
        run(test_env, True)
