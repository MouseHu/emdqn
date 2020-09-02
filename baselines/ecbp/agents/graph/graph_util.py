import tensorflow as tf
import baselines.common.tf_util as U
import numpy as np
import copy


def build_random_input(input_type,obs_shape):
    obs_input_augment = U.ensure_tf_input(input_type(obs_shape, None, name="obs_augment"))

    rand_img = tf.layers.conv2d(obs_input_augment.get(), 3, 3, padding='same',
                                kernel_initializer=tf.initializers.glorot_normal(), trainable=False, name='randcnn')

    init_rand_op = tf.variables_initializer([v for v in tf.global_variables() if 'randcnn' in v.name])

    augment_input_func = U.function(
            inputs=[obs_input_augment],
            outputs=[rand_img],
            updates=[]
        )
    rand_init_func= U.function([], [], updates=[init_rand_op])
    return augment_input_func,rand_init_func
