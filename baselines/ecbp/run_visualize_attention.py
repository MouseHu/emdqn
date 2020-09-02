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
from baselines.ecbp.agents.kbps_mp_agent import KBPSMPAgent
from baselines.ecbp.agents.ec_debug_agent import ECDebugAgent
from baselines.ecbp.agents.kbps_agent import KBPSAgent
from baselines.ecbp.agents.ec_agent import ECAgent
from baselines.ecbp.agents.human_agent import HumanAgent
from baselines.ecbp.agents.hybrid_agent import HybridAgent, HybridAgent2
from baselines.ecbp.agents.graph.build_graph_contrast_target import *
from baselines.ecbp.test.buffer_test import buffertest
import sys
import logging
from baselines.ecbp.agents.mer_bvae_attention_agent import BVAEAttentionAgent
sys.setrecursionlimit(30000)
# from gym.wrappers.monitoring.video_recorder import VideoRecorder

if __name__ == '__main__':

    args = parse_args()
    # if args.render:

    env = create_env(args)
    print("finish create env")
    exploration = PiecewiseSchedule([
        (0.1, 1),
    ], outside_value=0.1)

    try:
        num_actions = env.action_space.n
    except AttributeError:
        num_actions = env.unwrapped.pseudo_action_space.n

    obs_shape = env.observation_space.shape
    if obs_shape is None or obs_shape == (None,):
        obs_shape = env.unwrapped.observation_space.shape
    if obs_shape is None or obs_shape == (None,):
        obs_shape = env.unwrapped.observation_space['images'].shape
    if type(obs_shape) is int:
        obs_shape = (obs_shape,)

    ps_agent = MERAttentionAgent(
        representation_model_mlp if args.vector_input else representation_with_mask_model_cnn,
        # bvae_encoder, bvae_decoder,

        # representation_model_mlp if args.vector_input else representation_model_cnn,
        exploration,
        obs_shape, args.vector_input,
        args.lr,
        args.buffer_size, num_actions, args.latent_dim, args.gamma, args.knn,
        args.eval_epsilon, args.queue_threshold, args.batch_size, args.density, False, args.negative_samples,
        None)
    agent = ps_agent

    with U.make_session(4) as sess:
        # EMDQN

        U.initialize()
        saver = tf.train.Saver()
        num_iters, eval_iters, num_episodes = 0, 0, 0
        non_discount_return, discount_return = [0.0], [0.0]
        non_discount_return_eval, discount_return_eval = [0.0], [0.0]
        # Load the model
        start_time, start_steps = time.time(), 0
        eval_start_steps = 0
        steps_per_iter, iteration_time_est = RunningAvg(0.999, 1), RunningAvg(0.999, 1)
        print("before reset")
        obs = env.reset()
        agent.load(filedir=os.path.join(args.base_log_dir, args.load_dir), sess=sess, saver=saver, num_steps=args.num_steps)
        print("in main ", obs)
        print_flag = True
        # Main training loop
        train_time, act_time, env_time, update_time, cur_time = 0, 0, 0, 0, time.time()
        while True:
            eval = False
            if not eval:
                num_iters += 1
            else:
                eval_iters += 1
            # Take action and store transition in the replay buffer.
            action = agent.act(np.array(obs)[None], is_train=not eval)
            agent.save_attention(os.path.join(args.base_log_dir, args.load_dir), num_iters)
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
                non_discount_return_eval[-1] += rew
                discount_return_eval[-1] += rew * args.gamma ** (eval_iters - start_steps)
            else:
                non_discount_return[-1] += rew
                discount_return[-1] += rew * args.gamma ** (num_iters - start_steps)
            obs = new_obs
            if done:
                num_episodes += 1
                obs = env.reset()
                if eval:
                    non_discount_return_eval.append(0.0)
                    discount_return_eval.append(0.0)
                else:
                    non_discount_return.append(0.0)
                    discount_return.append(0.0)

            train_time += time.time() - cur_time
            cur_time = time.time()

            if start_time is not None:
                steps_per_iter.update(1)
                iteration_time_est.update(time.time() - start_time)
            start_time = time.time()

            if num_iters > 300 and done:
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
                logger.record_tabular("discount return_eval", np.mean(discount_return_eval[-return_len_eval - 1:-1]))
                logger.record_tabular("num episode", num_episodes)
                logger.record_tabular("update time", update_time)
                logger.record_tabular("train time", train_time)
                logger.record_tabular("act_time", act_time)
                logger.record_tabular("env_time", env_time)
                logger.record_tabular("eval", num_episodes % 2 == 0)

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
