import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np


def ib_model(img_in, noise, num_actions, scope, reuse=False, decoder="SPD"):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    img_size = img_in.shape[1]
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        coordinate = np.meshgrid(np.linspace(-1, 1, img_size), np.linspace(-1, 1, img_size))
        coordinate = tf.constant(coordinate)  # .to(torch.device("cuda"))

        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        z_mean = layers.fully_connected(out, num_outputs=512, activation_fn=None)
        z_logvar = layers.fully_connected(out, num_outputs=512, activation_fn=None)

        z = z_mean + noise * tf.exp(0.5 * z_logvar)

        with tf.variable_scope("action_value"):
            q_h = layers.fully_connected(z, num_outputs=512, activation_fn=tf.nn.relu)
            q_func = layers.fully_connected(q_h, num_outputs=num_actions, activation_fn=None)
        with tf.variable_scope("state_value"):

            v_h = layers.fully_connected(z, num_outputs=512, activation_fn=tf.nn.relu)
            v_mean = layers.fully_connected(v_h, num_outputs=1, activation_fn=None)
            v_logvar = layers.fully_connected(v_h, num_outputs=1, activation_fn=None)
        with tf.variable_scope("reconstruction"):
            if decoder == "DECONV":
                out = layers.fully_connected(z, num_outputs=196, activation_fn=tf.nn.relu)
                out = out.reshape(-1, 7, 7, 4)
                out = layers.conv2d_transpose(z, 32, kernel_size=3, strides=2, activation=tf.nn.relu, padding='same')
                out = layers.conv2d_transpose(z, 64, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same')
                reconstruct = layers.conv2d_transpose(z, 1, kernel_size=8, strides=3, activation=tf.nn.sigmoid,
                                                      padding='same')
            elif decoder == "SPD":
                out = layers.fully_connected(z, num_outputs=64, activation_fn=tf.nn.relu)
                out = out.reshape(-1, -1, 1, 1)
                out = tf.tile(out, [1, 1, img_size, img_size])
                out = layers.convolution2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=1, activation_fn=tf.nn.relu)
                reconstruct = layers.convolution2d(out, num_outputs=1, kernel_size=8, stride=1,
                                                   activation_fn=tf.nn.sigmoid)
                print(reconstruct.shape)
            else:
                print("Unrecognized decoder type.")
                raise NotImplementedError
        return q_func, v_mean, v_logvar, z_mean, z_logvar, reconstruct



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
