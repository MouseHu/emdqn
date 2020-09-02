from pyvirtualdisplay import Display

# display = Display(visible=1, size=(960, 640))
# display.start()
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
from baselines.ecbp.test.buffer_test import buffertest
import sys
import logging

sys.setrecursionlimit(30000)
# from gym.wrappers.monitoring.video_recorder import VideoRecorder

if __name__ == '__main__':

    args = parse_args()
    # if args.render:

    env = create_env(args)
    # env = GIFRecorder(video_path=args.video_path + "/{}/".format(args.comment), record_video=True, env=env)
    print("obs shape", env.observation_space.shape)
    subdir = (datetime.datetime.now()).strftime("%m-%d-%Y-%H:%M:%S") + " " + args.comment
    tfdir = os.path.join(args.base_log_dir, args.log_dir, subdir)
    tf_writer = tf.summary.FileWriter(tfdir, tf.get_default_graph())
    make_logger("ecbp", os.path.join(args.base_log_dir, args.log_dir, subdir, "logger.log"),stream_level=logging.DEBUG)
    make_logger("ec", os.path.join(args.base_log_dir, args.log_dir, subdir, "ec_logger.log"),stream_level=logging.DEBUG)
    exploration = PiecewiseSchedule([
        (0, 1),
        (args.end_training, 1.0),
        (args.end_training + 100000, 0.2),
        (args.end_training + 200000, 0.1),
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

    ps_agent = PSMPLearnTargetAgent(
        # unit_representation_model_mlp if args.vector_input else unit_representation_model_cnn,
        representation_model_mlp if args.vector_input else representation_model_cnn,
        exploration,
        obs_shape, args.vector_input,
        args.lr,
        args.buffer_size, num_actions, args.latent_dim, args.gamma, args.knn,
        args.eval_epsilon, args.queue_threshold, args.batch_size, args.density, args.trainable, args.negative_samples,
        tf_writer)

    human_agent = HumanAgent(
        {"w": 3, "s": 4, "d": 1, "a": 0, "x": 2, "p": 5, "3": 3, "4": 4, "1": 1, "0": 0, "2": 2, "5": 5})
    test_agent = HybridAgent2(ps_agent, human_agent, 30)
    agent = test_agent

    with U.make_session(4) as sess:
        # EMDQN

        U.initialize()
        saver = tf.train.Saver()
        num_iters = 0
        if args.load_dir is not None:
            agentdir = os.path.join(args.base_log_dir, args.agent_dir, args.load_dir)
            ps_agent.load(agentdir, sess, saver)
        while True:
            num_iters += 1
            action = agent.act(np.array(obs)[None], is_train=not eval)
            new_obs, rew, done, info = env.step(action)
            if args.render:
                env.render()

            agent.observe(action, rew, new_obs, done, train=False)

            # if num_iters % 10000 == 0:
            #     ps_agent.save(agentdir, sess, saver)

            obs = new_obs
            if done:
                obs = env.reset()

            if num_iters > args.num_steps and done:
                agent.finish()
                break
