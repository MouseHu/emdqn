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
    subdir = (datetime.datetime.now()).strftime("%m-%d-%Y-%H:%M:%S") + "_" + args.comment
    make_logger("ecbp", os.path.join(args.base_log_dir, args.load_dir, "ecbp_logger.log"),stream_level=logging.DEBUG)
    make_logger("ec", os.path.join(args.base_log_dir, args.load_dir, "ec_logger.log"),stream_level=logging.INFO)
    tfdir = os.path.join(args.base_log_dir, args.log_dir, subdir)
    agentdir = os.path.join(args.base_log_dir, args.agent_dir, subdir)
    tf_writer = tf.summary.FileWriter(tfdir, tf.get_default_graph())
    exploration = PiecewiseSchedule([
        (0.1, 1),
    ], outside_value=0.)

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
        debug=True,debug_dir=os.path.join(args.base_log_dir, args.load_dir),
        tf_writer=tf_writer)
    agent = ps_agent

    with U.make_session(4) as sess:
        # EMDQN

        U.initialize()
        saver = tf.train.Saver()
        # obs = env.reset()
        num_iters = 0
        ps_agent.load(filedir=os.path.join(args.base_log_dir, args.load_dir), sess=sess, saver=saver, num_steps=args.num_steps,load_model=False)
        agent.steps = 0
        while True:
            num_iters += 1
            agent.train()
            agent.steps += 1
            if num_iters > 100000:
                agent.finish()
                break
            if num_iters % 100 == 0:
                print("num iters: ",num_iters)
            if num_iters % 20000 == 0:
                print("begin testing attention")
                done = False
                eval_iters= 0
                obs = env.reset()
                while not done:
                    eval_iters += 1
                    agent.steps += 1
                    # Take action and store transition in the replay buffer.
                    action = agent.act(np.array(obs)[None], is_train=False)
                    agent.save_attention(agentdir, num_iters)
                    obs, rew, done, info = env.step(action)
                    agent.observe(action, rew, obs, done, train=False)
                agent.steps = num_iters