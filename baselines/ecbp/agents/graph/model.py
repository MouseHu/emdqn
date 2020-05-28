import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np


def ib_model(img_in, noise, num_actions, scope, reuse=False, decoder="DECONV"):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    img_size = img_in.shape[1]
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        # coordinate = np.meshgrid()
        coordinate = tf.meshgrid(np.linspace(-1, 1, img_size),
                                 np.linspace(-1, 1, img_size))  # .to(torch.device("cuda"))

        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        z_mean = layers.fully_connected(out, num_outputs=32, activation_fn=None)
        z_logvar = layers.fully_connected(out, num_outputs=32, activation_fn=None)

        z = z_mean + noise * tf.exp(0.5 * z_logvar)

        with tf.variable_scope("action_value"):
            q_h = layers.fully_connected(z, num_outputs=128, activation_fn=tf.nn.relu, scope="q_h_1",
                                         reuse=tf.AUTO_REUSE)
            q_h_deterministic = layers.fully_connected(z_mean, num_outputs=128, activation_fn=tf.nn.relu, scope="q_h_1",
                                                       reuse=True)
            q_func = layers.fully_connected(q_h, num_outputs=num_actions, activation_fn=None, scope="q_h_2",
                                            reuse=tf.AUTO_REUSE)
            q_func_deterministic = layers.fully_connected(q_h_deterministic, num_outputs=num_actions,
                                                          activation_fn=None, scope="q_h_2", reuse=True)
        with tf.variable_scope("state_value"):

            v_h = layers.fully_connected(z, num_outputs=128, activation_fn=tf.nn.relu)
            v_mean = layers.fully_connected(v_h, num_outputs=1, activation_fn=None)
            v_logvar = layers.fully_connected(v_h, num_outputs=1, activation_fn=None)
        with tf.variable_scope("reconstruction"):
            if decoder == "DECONV":
                out = layers.fully_connected(z, num_outputs=196, activation_fn=tf.nn.relu)
                out = tf.reshape(out, [-1, 7, 7, 4])
                out = layers.conv2d_transpose(out, 32, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                              padding='same')
                out = layers.conv2d_transpose(out, 64, kernel_size=4, stride=2, activation_fn=tf.nn.relu,
                                              padding='same')
                reconstruct = layers.conv2d_transpose(out, 4, kernel_size=8, stride=3, activation_fn=tf.nn.sigmoid,
                                                      padding='same')
                print(reconstruct.shape)
            elif decoder == "SPD":
                out = layers.fully_connected(z, num_outputs=64, activation_fn=tf.nn.relu)
                out = out.reshape(-1, -1, 1, 1)
                out = tf.tile(out, [1, 1, img_size, img_size])
                out = layers.convolution2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=1, activation_fn=tf.nn.relu)
                reconstruct = layers.convolution2d(out, num_outputs=4, kernel_size=8, stride=1,
                                                   activation_fn=tf.nn.sigmoid)
                print(reconstruct.shape)
            else:
                print("Unrecognized decoder type.")
                raise NotImplementedError
        return q_func, q_func_deterministic, v_mean, v_logvar, z_mean, z_logvar, reconstruct


def rp_model(img_in, num_actions, scope, reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    img_size = img_in.shape[1]
    print("img shape", img_in.shape)
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        out = layers.flatten(out)
        out = layers.fully_connected(out, num_outputs=32, activation_fn=None,
                                     weights_initializer=tf.random_normal_initializer(0, 1))
        return out, None


def modelbased_model_general(latent_in, action, num_actions, scope, reuse=False):
    latent_dim = int(latent_in.shape[1])
    # print(latent_in.shape)
    # print("huhao",type(latent_dim),latent_dim)
    action = tf.reshape(action, [-1, 1])
    with tf.variable_scope(scope, reuse=reuse):
        # action = tf.
        # action_in = tf.one_hot(action, num_actions)
        # input = tf.concat([layers.flatten(latent_in), layers.flatten(action_in)], axis=1)
        hidden = layers.fully_connected(latent_in, num_outputs=64, activation_fn=tf.nn.relu)
        hidden = layers.fully_connected(hidden, num_outputs=64, activation_fn=tf.nn.relu)
        hidden = layers.fully_connected(hidden, num_outputs=64, activation_fn=tf.nn.relu)
        latent_tp1 = layers.fully_connected(hidden, num_outputs=latent_dim * num_actions, activation_fn=None)
        latent_tp1 = tf.reshape(latent_tp1, [-1, num_actions, latent_dim])
        latent_tp1 = tf.gather(latent_tp1, action, axis=1, batch_dims=1)
        latent_tp1 = tf.reshape(latent_tp1, [-1, latent_dim])
        latent_tp1 = latent_tp1 + latent_in
        # latent_tp1 = latent_tp1 / tf.norm(latent_tp1)
        reward = layers.fully_connected(hidden, num_outputs=1, activation_fn=None)
    return latent_tp1, reward


def modelbased_model(latent_in, action, num_actions, scope, reuse=False):
    latent_dim = int(latent_in.shape[1])
    # print(latent_in.shape)
    # print("huhao",type(latent_dim),latent_dim)
    action = tf.reshape(action, [-1, 1])
    with tf.variable_scope(scope, reuse=reuse):
        # action = tf.
        # action_in = tf.one_hot(action, num_actions)
        # input = tf.concat([layers.flatten(latent_in), layers.flatten(action_in)], axis=1)
        hidden = layers.fully_connected(latent_in, num_outputs=64, activation_fn=tf.nn.relu)
        hidden = layers.fully_connected(hidden, num_outputs=64, activation_fn=tf.nn.relu)
        hidden = layers.fully_connected(hidden, num_outputs=64, activation_fn=tf.nn.relu)
        latent_tp1 = layers.fully_connected(hidden, num_outputs=latent_dim * num_actions, activation_fn=None)
        latent_tp1 = tf.reshape(latent_tp1, [-1, num_actions, latent_dim])
        latent_tp1 = tf.gather(latent_tp1, action, axis=1, batch_dims=1)
        latent_tp1 = tf.reshape(latent_tp1, [-1, latent_dim])
        latent_tp1 = latent_tp1 + latent_in
        # latent_tp1 = latent_tp1 / tf.norm(latent_tp1)
        reward = layers.fully_connected(hidden, num_outputs=1, activation_fn=None)
    return latent_tp1, reward


def contrastive_model(img_in, num_actions, scope, reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    img_size = img_in.shape[1]
    print("img shape", img_in.shape)
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        # coordinate = np.meshgrid()
        coordinate = tf.meshgrid(np.linspace(-1, 1, img_size),
                                 np.linspace(-1, 1, img_size))  # .to(torch.device("cuda"))

        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)

        z = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.tanh)
        normed_z = z / tf.norm(z, axis=1, keepdims=True)
        print("normed_z:", normed_z.shape)
        with tf.variable_scope("action_value"):
            v_h = layers.fully_connected(z, num_outputs=512, activation_fn=tf.nn.relu)
            v = layers.fully_connected(v_h, num_outputs=1, activation_fn=None)
        return normed_z, v


def contrastive_model_general(img_in, num_actions, scope, reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    img_size = img_in.shape[1]
    print("img shape", img_in.shape)
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        # coordinate = np.meshgrid()
        coordinate = tf.meshgrid(np.linspace(-1, 1, img_size),
                                 np.linspace(-1, 1, img_size))  # .to(torch.device("cuda"))

        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)

        z = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.tanh)
        # normed_z = z / tf.norm(z, axis=1, keepdims=True)
        # print("normed_z:", normed_z.shape)
        with tf.variable_scope("action_value"):
            v_h = layers.fully_connected(z, num_outputs=512, activation_fn=tf.nn.relu)
            v = layers.fully_connected(v_h, num_outputs=1, activation_fn=None)
        return z, v


def representation_model_cnn(img_in, num_actions, scope, reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    img_size = img_in.shape[1]
    print("img shape", img_in.shape)
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        # coordinate = np.meshgrid()

        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)

        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
        z = layers.fully_connected(out, num_outputs=32, activation_fn=None)
        # normed_z = z / tf.norm(z, axis=1, keepdims=True)
        # print("normed_z:", normed_z.shape)
        # with tf.variable_scope("action_value"):
        #     v_h = layers.fully_connected(z, num_outputs=512, activation_fn=tf.nn.relu)
        #     v = layers.fully_connected(v_h, num_outputs=1, activation_fn=None)
        return z


# def representation_model_mlp(obs_in, num_actions, scope, reuse=False):
#     """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
#     # obs_in = tf.reshape(obs_in, (-1,))
#     with tf.variable_scope(scope, reuse=reuse):
#         out = obs_in
#         # coordinate = np.meshgrid()
#
#         with tf.variable_scope("mlp"):
#             # original architecture
#             out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
#             out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
#             out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
#         out = layers.flatten(out)
#
#         out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
#         z = layers.fully_connected(out, num_outputs=32, activation_fn=None)
#         # normed_z = z / tf.norm(z, axis=1, keepdims=True)
#         # print("normed_z:", normed_z.shape)
#         # with tf.variable_scope("action_value"):
#         #     v_h = layers.fully_connected(z, num_outputs=512, activation_fn=tf.nn.relu)
#         #     v = layers.fully_connected(v_h, num_outputs=1, activation_fn=None)
#         return z

def representation_model_mlp(obs_in, num_actions, scope, reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    # obs_in = tf.reshape(obs_in, (-1,))
    with tf.variable_scope(scope, reuse=reuse):
        out = obs_in
        # coordinate = np.meshgrid()

        # with tf.variable_scope("mlp"):
        #     # original architecture
        #     out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
        #     out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
        #     out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
        # out = layers.flatten(out)
        #
        # out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
        z = layers.fully_connected(out, num_outputs=6, activation_fn=None)
        # normed_z = z / tf.norm(z, axis=1, keepdims=True)
        # print("normed_z:", normed_z.shape)
        # with tf.variable_scope("action_value"):
        #     v_h = layers.fully_connected(z, num_outputs=512, activation_fn=tf.nn.relu)
        #     v = layers.fully_connected(v_h, num_outputs=1, activation_fn=None)
        return z


def dueling_model(img_in, num_actions, scope, reuse=False):
    """As described in https://arxiv.org/abs/1511.06581"""
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)

        with tf.variable_scope("state_value"):
            state_hidden = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
            state_score = layers.fully_connected(state_hidden, num_outputs=1, activation_fn=None)
        with tf.variable_scope("action_value"):
            actions_hidden = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(actions_hidden, num_outputs=num_actions, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores = action_scores - tf.expand_dims(action_scores_mean, 1)

        return state_score + action_scores


def model(img_in, num_actions, scope, reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out


def simhash_model(img_in, scope, decoder='SPD', reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        img_size = img_in.shape[1]
        with tf.variable_scope("encoder"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        code = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.sigmoid)
        with tf.variable_scope("reconstruction"):
            if decoder == "DECONV":
                out = layers.fully_connected(code, num_outputs=196, activation_fn=tf.nn.relu)
                out = tf.reshape(out, [-1, 7, 7, 4])
                out = layers.conv2d_transpose(out, 32, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                              padding='same')
                out = layers.conv2d_transpose(out, 64, kernel_size=4, stride=2, activation_fn=tf.nn.relu,
                                              padding='same')
                reconstruct = layers.conv2d_transpose(out, 4, kernel_size=8, stride=3, activation_fn=tf.nn.sigmoid,
                                                      padding='same')
                print(reconstruct.shape)
            elif decoder == "SPD":
                out = layers.fully_connected(code, num_outputs=64, activation_fn=tf.nn.relu)
                out = out.reshape(-1, -1, 1, 1)
                out = tf.tile(out, [1, 1, img_size, img_size])
                out = layers.convolution2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=1, activation_fn=tf.nn.relu)
                reconstruct = layers.convolution2d(out, num_outputs=4, kernel_size=8, stride=1,
                                                   activation_fn=tf.nn.sigmoid)
                print(reconstruct.shape)
            else:
                print("Unrecognized decoder type.")
                raise NotImplementedError
        return code, reconstruct


def ib_dueling_model(img_in, num_actions, scope, reuse=False):
    """As described in https://arxiv.org/abs/1511.06581"""
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)

        with tf.variable_scope("state_value"):
            state_hidden = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
            state_score = layers.fully_connected(state_hidden, num_outputs=1, activation_fn=None)
        with tf.variable_scope("action_value"):
            actions_hidden = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(actions_hidden, num_outputs=num_actions, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores = action_scores - tf.expand_dims(action_scores_mean, 1)

        return state_score + action_scores
