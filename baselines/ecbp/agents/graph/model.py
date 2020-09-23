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

        normed_z = z / tf.maximum(1e-10, tf.norm(z, axis=1, keepdims=True))

        # print("normed_z:", normed_z.shape)
        # with tf.variable_scope("action_value"):
        #     v_h = layers.fully_connected(z, num_outputs=512, activation_fn=tf.nn.relu)
        #     v = layers.fully_connected(v_h, num_outputs=1, activation_fn=None)
        return normed_z


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


def representation_model_mlp(obs_in, num_actions, scope, reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    # obs_in = tf.reshape(obs_in, (-1,))
    with tf.variable_scope(scope, reuse=reuse):
        out = obs_in
        # coordinate = np.meshgrid()

        with tf.variable_scope("mlp"):
            #     # original architecture
            out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        #
        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
        z = layers.fully_connected(out, num_outputs=32, activation_fn=None)

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


def unit_representation_model_cnn(img_in, num_actions, scope, reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    img_size = img_in.shape[1]
    print("img shape", img_in.shape)
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        # coordinate = np.meshgrid()

        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.leaky_relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.leaky_relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.leaky_relu)
        out = layers.flatten(out)

        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.leaky_relu)
        z = layers.fully_connected(out, num_outputs=32, activation_fn=None)
        normed_z = z / tf.maximum(1e-10, tf.norm(z, axis=1, keepdims=True))
        # normed_z =
        # print("normed_z:", normed_z.shape)
        # with tf.variable_scope("action_value"):
        #     v_h = layers.fully_connected(z, num_outputs=512, activation_fn=tf.nn.relu)
        #     v = layers.fully_connected(v_h, num_outputs=1, activation_fn=None)
        return normed_z


def representation_with_mask_model_cnn(img_in, num_actions, scope, reuse=False, unit=True):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    img_size = img_in.shape[1]
    batch_size = img_in.shape[0]
    print("img shape", img_in.shape)
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        # coordinate = np.meshgrid()

        with tf.variable_scope("convnet"):
            # original architecture
            # out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.leaky_relu,
            #                            padding='same')
            # out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.leaky_relu,
            #                            padding='same')
            # out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.leaky_relu,
            #                            padding='same')
            out = tf.pad(out, tf.constant([[0, 0], [4, 4], [4, 4], [0, 0]]), "REFLECT")
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.leaky_relu,
                                       padding='valid')
            out = tf.pad(out, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]), "REFLECT")
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.leaky_relu,
                                       padding='valid')
            out = tf.pad(out, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "REFLECT")
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.leaky_relu,
                                       padding='valid')

        # spatial attention
        out_max = tf.reduce_max(out, axis=3, keepdims=True)
        out_mean = tf.reduce_mean(out, axis=3, keepdims=True)
        # out_min = tf.reduce_min(out, axis=3, keepdims=True)
        # out_mean = tf.reduce_mean(out, axis=3, keepdims=True)
        attention_feature = tf.concat([out_max, out_mean], axis=3)
        # print("attention shape ", attention_feature.shape)
        # attention_latent = layers.convolution2d(attention_feature, num_outputs=32, kernel_size=1, stride=1,
        #                                         activation_fn=tf.nn.relu)
        attention = layers.convolution2d(attention_feature, num_outputs=1, kernel_size=1, stride=1,
                                         activation_fn=tf.nn.sigmoid)

        attention_max = tf.reduce_max(tf.reduce_max(attention, axis=1, keep_dims=True), axis=2, keep_dims=True)
        attention_min = tf.reduce_min(tf.reduce_min(attention, axis=1, keep_dims=True), axis=1, keep_dims=True)
        attention_normalized = (attention - attention_min) / (attention_max - attention_min + 1e-9)
        soft_out = tf.multiply(attention, out)
        # out = attention_normalized
        soft_out = layers.flatten(soft_out)
        hard_out = tf.multiply(attention_normalized, out)
        hard_out = layers.flatten(hard_out)
        # value_latent = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
        # value_latent = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.leaky_relu)
        # with tf.variable_scope("value_regression"):
        value = layers.fully_connected(soft_out, num_outputs=num_actions, activation_fn=None)

        out_z = layers.fully_connected(hard_out, num_outputs=32, activation_fn=tf.nn.leaky_relu)
        z = layers.fully_connected(out_z, num_outputs=32, activation_fn=None)

        # normalized_out = layers.flatten(normalized_out)
        # hash = layers.fully_connected(normalized_out, num_outputs=32, activation_fn=None)
        # print("???",hash.shape)
        if unit:
            contrast_z = z / tf.maximum(1e-10, tf.norm(z, axis=1, keepdims=True))
        return attention, value, contrast_z, contrast_z
        # return contrast_z


def unit_representation_model_mlp(obs_in, num_actions, scope, reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    # obs_in = tf.reshape(obs_in, (-1,))
    with tf.variable_scope(scope, reuse=reuse):
        out = obs_in
        attention = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.sigmoid)
        out = tf.multiply(attention, out)
        value_latent = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.leaky_relu)
        value = layers.fully_connected(value_latent, num_outputs=1, activation_fn=None)
        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.leaky_relu)
        z = layers.fully_connected(out, num_outputs=32, activation_fn=None)
        normed_z = z / tf.norm(z, axis=1, keepdims=True)
        return attention, value, normed_z


def unit_representation_with_mask_model_mlp(obs_in, num_actions, scope, reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    # obs_in = tf.reshape(obs_in, (-1,))
    with tf.variable_scope(scope, reuse=reuse):
        out = obs_in
        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.leaky_relu)
        z = layers.fully_connected(out, num_outputs=32, activation_fn=None)
        normed_z = z / tf.norm(z, axis=1, keepdims=True)
        return normed_z


def dbc_model_mlp(obs_in, num_actions, scope, reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    # obs_in = tf.reshape(obs_in, (-1,))
    latent_dim = obs_in.shape[-1]
    with tf.variable_scope(scope, reuse=reuse):
        out = obs_in
        # coordinate = np.meshgrid()

        with tf.variable_scope("mlp"):
            # original architecture
            out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
        # out = layers.flatten(out)
        #
        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
        z_predict = layers.fully_connected(out, num_outputs=num_actions * latent_dim, activation_fn=None)
        r_predict = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return z_predict, r_predict


def bvae_encoder(obs_in, noise, scope, reuse=False, use_mlp=False, normalize=False, use_mask=True):
    latent_dim = obs_in.shape[-1]
    with tf.variable_scope(scope, reuse=reuse):
        out = obs_in
        if use_mlp:
            out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.leaky_relu)
            h_mean = layers.fully_connected(out, num_outputs=32, activation_fn=None)
            h_logvar = layers.fully_connected(out, num_outputs=32, activation_fn=None)
            h = h_mean + noise * tf.exp(0.5 * h_logvar)

        else:
            with tf.variable_scope("convnet"):
                # original architecture
                out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.leaky_relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.leaky_relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.leaky_relu)
            out = layers.flatten(out)
            h_mean = layers.fully_connected(out, num_outputs=32, activation_fn=None)
            h_logvar = layers.fully_connected(out, num_outputs=32, activation_fn=None)
            h = h_mean + noise * tf.exp(0.5 * h_logvar)
            origin_z = tf.concat([h_mean, h_logvar], axis=1)

            if use_mask:
                attention = layers.fully_connected(origin_z, num_outputs=64, activation_fn=tf.nn.sigmoid)
                z = tf.multiply(origin_z, attention)
            else:
                z = origin_z
            # if normalize:
            #     z = z / tf.norm(z, axis=1, keepdims=True)
        return z, h, h_mean, h_logvar, attention, origin_z


def bvae_decoder(h, scope, reuse=False, use_mlp=False):
    with tf.variable_scope(scope, reuse=reuse):
        if use_mlp:
            decode_out = layers.fully_connected(h, num_outputs=32, activation_fn=tf.nn.leaky_relu)
            reconstruct = layers.fully_connected(decode_out, num_outputs=32, activation_fn=None)

        else:
            with tf.variable_scope("reconstruction"):
                decode_out = layers.fully_connected(h, num_outputs=196, activation_fn=tf.nn.relu)
                decode_out = tf.reshape(decode_out, [-1, 7, 7, 4])
                decode_out = layers.conv2d_transpose(decode_out, 32, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                                     padding='same')  # 14*14
                decode_out = layers.conv2d_transpose(decode_out, 32, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                                     padding='same')  # 28*28
                decode_out = layers.conv2d_transpose(decode_out, 32, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                                     padding='same')  # 56*56
                decode_out = layers.conv2d_transpose(decode_out, 64, kernel_size=4, stride=2, activation_fn=tf.nn.relu,
                                                     padding='same')  # 112*112
                reconstruct = layers.conv2d_transpose(decode_out, 3, kernel_size=8, stride=2,
                                                      activation_fn=tf.nn.sigmoid,
                                                      padding='same')  # 224*224
        value_latent = layers.fully_connected(h, num_outputs=32, activation_fn=tf.nn.leaky_relu)
        value = layers.fully_connected(value_latent, num_outputs=1, activation_fn=None)
        # z = layers.fully_connected(h, num_outputs=32, activation_fn=None)

        return reconstruct, value


def mer_bvae_model(obs_in, noise, scope, reuse=False, use_mlp=False, normalize=False, use_mask=True):
    latent_dim = obs_in.shape[-1]
    with tf.variable_scope(scope, reuse=reuse):
        out = obs_in
        if use_mlp:
            out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.leaky_relu)
            h_mean = layers.fully_connected(out, num_outputs=32, activation_fn=None)
            h_logvar = layers.fully_connected(out, num_outputs=32, activation_fn=None)
            h = h_mean + noise * tf.exp(0.5 * h_logvar)
            decode_out = layers.fully_connected(h, num_outputs=32, activation_fn=tf.nn.leaky_relu)
            reconstruct = layers.fully_connected(decode_out, num_outputs=32, activation_fn=None)

        else:
            with tf.variable_scope("convnet"):
                # original architecture
                out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.leaky_relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.leaky_relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.leaky_relu)
            # print(out.shape)
            out = layers.flatten(out)
            # print(out.shape)
            h_mean = layers.fully_connected(out, num_outputs=32, activation_fn=None)
            # print(h_mean.shape)
            h_logvar = layers.fully_connected(out, num_outputs=32, activation_fn=None)
            # print(out.shape)
            print("?", h_mean.shape, noise.shape, obs_in.shape)
            h = h_mean + noise * tf.exp(0.5 * h_logvar)
            with tf.variable_scope("reconstruction"):
                decode_out = layers.fully_connected(h, num_outputs=196, activation_fn=tf.nn.relu)
                decode_out = tf.reshape(decode_out, [-1, 7, 7, 4])
                decode_out = layers.conv2d_transpose(decode_out, 32, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                                     padding='same')  # 14*14
                decode_out = layers.conv2d_transpose(decode_out, 32, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                                     padding='same')  # 28*28
                decode_out = layers.conv2d_transpose(decode_out, 32, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                                     padding='same')  # 56*56
                decode_out = layers.conv2d_transpose(decode_out, 64, kernel_size=4, stride=2, activation_fn=tf.nn.relu,
                                                     padding='same')  # 112*112
                reconstruct = layers.conv2d_transpose(decode_out, 3, kernel_size=8, stride=3,
                                                      activation_fn=tf.nn.sigmoid,
                                                      padding='same')  # 224*224
        value_latent = layers.fully_connected(h, num_outputs=32, activation_fn=tf.nn.leaky_relu)
        value = layers.fully_connected(value_latent, num_outputs=1, activation_fn=None)
        # z = layers.fully_connected(h, num_outputs=32, activation_fn=None)
        z = tf.concat([h_mean, h_logvar], axis=1)
        if use_mask:
            attention = layers.fully_connected(z, num_outputs=64, activation_fn=tf.nn.sigmoid)
            z = tf.multiply(z, attention)
        if normalize:
            z = z / tf.norm(z, axis=1, keepdims=True)
        return z, h_mean, h_logvar, attention, reconstruct, value
