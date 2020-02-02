"""Deep Q learning graph

The functions in this file can are used to create the following functions:

======= act ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative not update happens
        (default: no update)

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= train =======

    Function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:

        td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
        loss = huber_loss[td_error]

    Parameters
    ----------
    obs_t: object
        a batch of observations
    action: np.array
        actions that were selected upon seeing obs_t.
        dtype must be int32 and shape must be (batch_size,)
    reward: np.array
        immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,)
    obs_tp1: object
        observations that followed obs_t
    done: np.array
        1 if obs_t was the last observation in the episode and 0 otherwise
        obs_tp1 gets ignored, but must be of the valid shape.
        dtype must be float32 and shape must be (batch_size,)
    weight: np.array
        imporance weights for every element of the batch (gradient is multiplied
        by the importance weight) dtype must be float32 and shape must be (batch_size,)

    Returns
    -------
    td_error: np.array
        a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)

======= update_target ========

    copy the parameters from optimized Q function to the target Q function.
    In Q learning we actually optimize the following error:

        Q(s,a) - (r + gamma * max_a' Q'(s', a'))

    Where Q' is lagging behind Q to stablize the learning. For example for Atari

    Q' is set to Q once every 10000 updates training steps.

"""
import tensorflow as tf
import baselines.common.tf_util as U
import numpy as np


def build_act_mfmc(make_obs_ph, model_func, num_actions, scope="deepq", secondary_scope="model_func", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        z, v = model_func(observations_ph.get(),
                          num_actions,
                          scope=secondary_scope,
                          reuse=tf.AUTO_REUSE)

        act = U.function(inputs=[observations_ph],
                         outputs=[z])

        return act


def build_train_mfmc(make_obs_ph, model_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0, scope="mfec",
                     latent_dim=32, input_dim=84 * 84 * 4, hash_dim=32, K=10, beta=0.1, predict=True, reuse=None):
    """Creates the train function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
    num_actions: int
        number of actions
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    grad_norm_clipping: float or None
        clip gradient norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """
    z_func = build_act_mfmc(make_obs_ph, model_func, num_actions, scope=scope, secondary_scope="model_func",
                            reuse=reuse)
    encoder_z_func = build_act_mfmc(make_obs_ph, model_func, num_actions, scope=scope,
                                    secondary_scope="encoder_model_func", reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders

        # EMDQN
        tau = tf.placeholder(tf.float32, [1], name='tau')
        momentum = tf.placeholder(tf.float32, [1], name='momentum')

        obs_hash_input = U.ensure_tf_input(make_obs_ph("obs_hash"))
        obs_mc_input = U.ensure_tf_input(make_obs_ph("obs"))
        obs_mc_input_query = U.ensure_tf_input(make_obs_ph("obs_query"))
        obs_mc_input_positive = U.ensure_tf_input(make_obs_ph("enc_obs_pos"))
        keys_mc_input_negative = tf.placeholder(tf.float32, [None, K, latent_dim], name='enc_keys_neg')

        # inputs = [obs_mc_input]
        value_input = tf.placeholder(tf.float32, [None, 1], name='value')
        if predict:
            inputs = [tau, obs_mc_input_query, obs_mc_input_positive, keys_mc_input_negative, obs_mc_input, value_input]
        else:
            inputs = [tau, obs_mc_input_query, obs_mc_input_positive, keys_mc_input_negative]
        z_mc, _ = model_func(
            obs_mc_input_query.get(), num_actions,
            scope="model_func",
            reuse=True)

        _, v_mc = model_func(
            obs_mc_input.get(), num_actions,
            scope="model_func",
            reuse=True)
        encoder_z_mc_pos, encoder_v_mc_pos = model_func(
            obs_mc_input_positive.get(), num_actions,
            scope="encoder_model_func", reuse=True)

        z_mc = tf.reshape(z_mc, [-1, latent_dim, 1])
        # z_mc_tile = tf.tile(z_mc_expand, [1, K, 1])
        negative = tf.reduce_sum(tf.exp(tf.matmul(keys_mc_input_negative, z_mc) / tau), axis=1)
        positive = tf.reduce_sum(z_mc * encoder_z_mc_pos, 1) / tau
        print("shape:", z_mc.shape, encoder_z_mc_pos.shape, negative.shape, positive.shape)
        contrast_loss = tf.reduce_sum(tf.log(negative) - positive)
        # print("shape2:", z_mc.shape, negative.shape, positive.shape)
        prediction_loss = tf.losses.mean_squared_error(value_input, v_mc)
        total_loss = contrast_loss
        if predict:
            total_loss += beta * prediction_loss

        model_func_vars = U.scope_vars(U.absolute_scope_name("model_func"))
        encoder_model_func_vars = U.scope_vars(U.absolute_scope_name("encoder_model_func"))
        if grad_norm_clipping is not None:
            optimize_expr_contrast_with_prediction = U.minimize_and_clip(optimizer,
                                                                         total_loss,
                                                                         var_list=model_func_vars,
                                                                         clip_val=grad_norm_clipping)
        else:
            optimize_expr_contrast_with_prediction = optimizer.minimize(total_loss, var_list=model_func_vars)
        # Create callable functions
        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(model_func_vars, key=lambda v: v.name),
                                   sorted(encoder_model_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign((1 - momentum) * var + momentum * var_target))
        update_target_expr = tf.group(*update_target_expr)
        update_target = U.function([momentum], [], updates=[update_target_expr])

        latten_obs = tf.reshape(obs_hash_input.get(), [-1, input_dim])
        rp = tf.random.normal([input_dim, hash_dim], 0, 1 / np.sqrt(hash_dim))
        obs_hash_output = tf.matmul(latten_obs, rp)
        hash_func = U.function(
            inputs=[obs_hash_input],
            outputs=[obs_hash_output]
        )
        # EMDQN

        z_norm_summary = tf.summary.scalar("z_norm", tf.reduce_mean(tf.norm(z_mc, axis=1)))
        encoder_z_norm_summary = tf.summary.scalar("z_norm", tf.reduce_mean(tf.norm(encoder_z_mc_pos, axis=1)))
        neg_norm_summary = tf.summary.scalar("z_norm", tf.reduce_mean(tf.norm(keys_mc_input_negative, axis=[1, 2])))
        contrast_loss_summary = tf.summary.scalar("contrast loss", tf.reduce_mean(contrast_loss))
        prediction_loss_summary = tf.summary.scalar("prediction loss", tf.reduce_mean(prediction_loss))
        total_loss_summary = tf.summary.scalar("total loss", tf.reduce_mean(total_loss))

        if predict:
            summaries = [z_norm_summary, encoder_z_norm_summary, neg_norm_summary, contrast_loss_summary,
                         prediction_loss_summary, total_loss_summary]
        else:
            summaries = [z_norm_summary, encoder_z_norm_summary, neg_norm_summary, contrast_loss_summary,
                         total_loss_summary]
        summary = tf.summary.merge(summaries)

        train = U.function(
            inputs=inputs,
            outputs=[total_loss, summary],
            updates=[optimize_expr_contrast_with_prediction]
        )

        return hash_func, z_func, encoder_z_func, update_target, train
