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


def build_act_modelbased(make_obs_ph, net_func, num_actions, scope="deepq", secondary_scope="model_func", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        z, v = net_func(observations_ph.get(),
                        num_actions,
                        scope=secondary_scope,
                        reuse=tf.AUTO_REUSE)

        act = U.function(inputs=[observations_ph],
                         outputs=[z])

        return act


def build_train_modelbased(make_obs_ph, net_func, model_func, num_actions, optimizer, grad_norm_clipping=None,
                           gamma=1.0,
                           scope="mfec",
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

    z_func = build_act_modelbased(make_obs_ph, net_func, num_actions, scope=scope, secondary_scope="net_func",
                                  reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders

        # EMDQN
        tau = tf.placeholder(tf.float32, [1], name='tau')
        # momentum = tf.placeholder(tf.float32, [1], name='momentum')

        obs_mc_input_query = U.ensure_tf_input(make_obs_ph("obs_query"))
        obs_mc_input_positive = U.ensure_tf_input(make_obs_ph("enc_obs_pos"))
        obs_mc_input_negative = U.ensure_tf_input(make_obs_ph("enc_obs_neg"))
        obs_mc_input_model_t = U.ensure_tf_input(make_obs_ph("obs_query"))
        obs_mc_input_model_tp1 = U.ensure_tf_input(make_obs_ph("obs_query"))
        reward_input_model = tf.placeholder(tf.float32, [None], name='reward')
        action_input_model = tf.placeholder(tf.int32, [None], name='action')
        latent_input_out = tf.placeholder(tf.float32, [None, latent_dim], name='latent')
        action_input_out = tf.placeholder(tf.int32, [None], name='action_input_out')
        # inputs = [obs_mc_input]
        # inputs = [tau, obs_mc_input_query, obs_mc_input_positive, obs_mc_input_negative]
        inputs = [tau, obs_mc_input_query, obs_mc_input_positive, obs_mc_input_negative, obs_mc_input_model_t,
                  obs_mc_input_model_tp1, reward_input_model, action_input_model]
        z_mc_model_t, _ = net_func(
            obs_mc_input_model_t.get(), num_actions,
            scope="net_func",
            reuse=True)
        z_mc_model_tp1, _ = net_func(
            obs_mc_input_model_tp1.get(), num_actions,
            scope="net_func",
            reuse=True)
        z_mc_out, reward_out = model_func(latent_input_out,
                                          action_input_out, num_actions, scope="model_func",
                                          reuse=reuse)
        z_mc_model_tp1_predict, reward_predict = model_func(z_mc_model_t,
                                                            action_input_model, num_actions, scope="model_func",
                                                            reuse=True)
        z_mc, _ = net_func(
            obs_mc_input_query.get(), num_actions,
            scope="net_func",
            reuse=True)

        # _, v_mc = net_func(
        #     obs_mc_input_query.get(), num_actions,
        #     scope="net_func",
        #     reuse=True)
        z_mc_pos, v_mc_pos = net_func(
            obs_mc_input_positive.get(), num_actions,
            scope="net_func", reuse=True)

        z_mc_neg, v_mc_neg = net_func(
            obs_mc_input_negative.get(), num_actions,
            scope="net_func", reuse=True)

        z_mc_pos = tf.reshape(z_mc_pos, [-1, 1, latent_dim])
        z_mc = tf.reshape(z_mc, [-1, latent_dim, 1])
        z_mc_neg = tf.reshape(z_mc_neg, [-1, K, latent_dim])

        negative = tf.matmul(z_mc_neg, z_mc) / tau
        sum_negative = tf.squeeze(tf.reduce_sum(tf.exp(negative), axis=1))
        positive = tf.squeeze(tf.matmul(z_mc_pos, z_mc) / tau)
        print("shape:", z_mc.shape, z_mc_pos.shape, z_mc_neg.shape, sum_negative.shape, negative.shape,
              positive.shape)
        contrast_loss = tf.reduce_mean(tf.log(sum_negative) - positive)
        # # print("shape2:", z_mc.shape, negative.shape, positive.shape)
        # # prediction_loss = tf.losses.mean_squared_error(value_input, v_mc)
        # total_loss = contrast_loss
        # if predict:
        #     total_loss += beta * prediction_loss

        model_func_vars = U.scope_vars(U.absolute_scope_name("model_func")) + U.scope_vars(
            U.absolute_scope_name("net_func"))
        # encoder_net_func_vars = U.scope_vars(U.absolute_scope_name("encoder_net_func"))

        transition_loss = tf.reduce_sum(tf.square(z_mc_model_tp1 -z_mc_model_tp1_predict))
        reward_loss = tf.reduce_sum(tf.square(reward_predict - reward_input_model))
        total_loss = contrast_loss + transition_loss + reward_loss
        if grad_norm_clipping is not None:
            optimize_expr_contrast_with_prediction = U.minimize_and_clip(optimizer,
                                                                         total_loss,
                                                                         var_list=model_func_vars,
                                                                         clip_val=grad_norm_clipping)
        else:
            optimize_expr_contrast_with_prediction = optimizer.minimize(total_loss, var_list=model_func_vars)
        # Create callable functions
        # update_target_fn will be called periodically to copy Q network to target Q network
        z_var_summary = tf.summary.scalar("z_var", tf.reduce_mean(tf.math.reduce_std(z_mc_model_t, axis=1)))
        negative_summary = tf.summary.scalar("negative", tf.reduce_mean(tf.reduce_mean(negative)))
        positive_summary = tf.summary.scalar("positive", tf.reduce_mean(tf.reduce_mean(positive)))
        contrast_loss_summary = tf.summary.scalar("contrast loss", tf.reduce_mean(contrast_loss))
        transition_loss_summary = tf.summary.scalar("transition loss", tf.reduce_mean(transition_loss))
        trivial_loss_summary = tf.summary.scalar("trivial loss",
                                                 tf.reduce_mean(tf.square(z_mc_model_t - z_mc_model_tp1)))
        reward_loss_summary = tf.summary.scalar("reward loss", tf.reduce_mean(reward_loss))
        # prediction_loss_summary = tf.summary.scalar("prediction loss", tf.reduce_mean(prediction_loss))
        total_loss_summary = tf.summary.scalar("total loss", tf.reduce_mean(total_loss))

        summaries = [z_var_summary, negative_summary, positive_summary, contrast_loss_summary, trivial_loss_summary,
                     transition_loss_summary, reward_loss_summary, total_loss_summary]
        summary = tf.summary.merge(summaries)

        train = U.function(
            inputs=inputs,
            outputs=[total_loss, summary],
            updates=[optimize_expr_contrast_with_prediction]
        )
        prediction = U.function(
            inputs=[latent_input_out, action_input_out],
            outputs=[z_mc_out, reward_out]
        )
        return z_func, prediction, train
