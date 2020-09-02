import tensorflow as tf
from pyvirtualdisplay import Display
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
display = Display(visible=1, size=(640, 480))
display.start()
from baselines.ecbp.util import *
from baselines.ecbp.agents.graph.build_graph_contrast_target import *
from baselines.ecbp.test.value_iteration import *
args = parse_args()
    # if args.render:

env = create_env(args)
num_iters = 900900
obs_shape = env.observation_space.shape
if obs_shape is None or obs_shape == (None,):
    obs_shape = env.unwrapped.observation_space.shape
if type(obs_shape) is int:
    obs_shape = (obs_shape,)

try:
    num_actions = env.action_space.n
except AttributeError:
    num_actions = env.unwrapped.pseudo_action_space.n
input_type = U.Float32Input if args.vector_input else U.Uint8Input
loss_type = ["contrast"]
with U.make_session(4) as sess:
    hash_func, _, _, _, _ = build_train_contrast_target(
        make_obs_ph=lambda name: input_type(obs_shape, name=name),
        model_func=representation_model_mlp if args.vector_input else representation_model_cnn,
        num_actions=num_actions,
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-4),
        gamma=args.gamma,
        grad_norm_clipping=10,
        latent_dim=args.latent_dim,
        loss_type=loss_type
    )
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(args.base_log_dir, args.log_dir, "./model/model{}_{}.ckpt".format(args.comment,args.num_steps)))
    with tf.variable_scope("mfec", reuse=True):
        magic_num = tf.get_variable("magic")
        # sess.run(magic_num.assign([142857]))
        print(magic_num.eval())

    values = value_iteration(env,args.gamma)
    zs = np.zeros((obs_shape[0],args.latent_dim))
    for i in range(obs_shape[0]):
        one_hot = np.zeros(obs_shape[0])
        one_hot[i] = 1
        zs[i] = np.array(hash_func(one_hot[np.newaxis,...])).squeeze()

    tsne = TSNE(n_components=2,random_state=2)
    x_embeded = tsne.fit_transform(zs)

    plt.figure(figsize=(8, 8))
    plt.scatter(x_embeded[:,0],x_embeded[:,1],c= values,label=np.arange(obs_shape[0]))
    plt.colorbar()
    for i,coord in enumerate(x_embeded):
        x,y = coord
        plt.annotate(
            "{}".format(i),
            xy=(x, y),
            textcoords='offset points',
            ha='center',
            va='top')
    plt.show()