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


def build_act_contrast(make_obs_ph, model_func, num_actions, scope="deepq", secondary_scope="model_func", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        z, v = model_func(observations_ph.get(),
                          num_actions,
                          scope=secondary_scope,
                          reuse=tf.AUTO_REUSE)

        act = U.function(inputs=[observations_ph],
                         outputs=[z])

        return act


def emb_dist(a, b):
    return tf.maximum(0., tf.reduce_sum(tf.squared_difference(a, b), axis=1))


def contrastive_loss_fc(emb_cur, emb_next, emb_neq, margin=1, c_type='origin'):
    if c_type is None or c_type == 'origin':
        return tf.reduce_mean(
            tf.maximum(emb_dist(emb_cur, emb_next) - emb_dist(emb_cur, emb_neq) + margin, 0))
    elif c_type == 'sqmargin':
        return tf.reduce_mean(emb_dist(emb_cur, emb_next) +
                              tf.maximum(0.,
                                         margin - emb_dist(emb_cur, emb_neq)))
    else:
        return tf.reduce_mean(emb_dist(emb_cur, emb_next) + tf.square(tf.maximum(0., margin -
                                                                                 tf.math.sqrt(
                                                                                     emb_dist(emb_cur,
                                                                                              emb_neq)))))


def build_train_contrast(make_obs_ph, model_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0,
                         scope="mfec",
                         latent_dim=32, alpha=0.05, beta=0.1, theta=0.1, loss_type=["contrast"], c_loss_type="sqmargin",
                         reuse=None):
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

    # z_func = build_act_contrast(make_obs_ph, model_func, num_actions, scope=scope, secondary_scope="model_func",
    #                             reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders

        # EMDQN
        # tau = tf.placeholder(tf.float32, [1], name='tau')
        # momentum = tf.placeholder(tf.float32, [1], name='momentum')

        obs_input_query = U.ensure_tf_input(make_obs_ph("obs_query"))
        obs_input_positive = U.ensure_tf_input(make_obs_ph("enc_obs_pos"))
        obs_input_negative = U.ensure_tf_input(make_obs_ph("enc_obs_neg"))

        value_input_query = tf.placeholder(tf.float32, [None], name="value")
        action_embedding = tf.Variable(tf.random_normal([num_actions, latent_dim], stddev=1), name="action_embedding")
        action_input = tf.placeholder(tf.int32, [None], name="action")
        inputs = [obs_input_query]
        if "contrast" in loss_type:
            inputs += [obs_input_positive, obs_input_negative]
        if "regression" in loss_type:
            inputs += [value_input_query]
        if "linear_model" in loss_type:
            inputs += [action_input]
            if "contrast" not in loss_type:
                inputs += [obs_input_positive]
        z = model_func(
            obs_input_query.get(), num_actions,
            scope="model_func",
            reuse=tf.AUTO_REUSE)

        h = model_func(
            obs_input_query.get(), num_actions,
            scope="hash_func",
            reuse=False)

        # _, v = model_func(
        #     obs_input_query.get(), num_actions,
        #     scope="model_func",
        #     reuse=True)
        z_pos = model_func(
            obs_input_positive.get(), num_actions,
            scope="model_func", reuse=True)

        z_neg = model_func(
            obs_input_negative.get(), num_actions,
            scope="model_func", reuse=True)

        z_pos = tf.reshape(z_pos, [-1, latent_dim])
        z_tar = tf.reshape(z, [-1, latent_dim])
        z_neg = tf.reshape(z_neg, [-1, latent_dim])

        contrast_loss = contrastive_loss_fc(z_tar, z_pos, z_neg, c_type=c_loss_type)

        regression_loss = tf.reduce_mean(tf.squared_difference(tf.norm(z_tar,axis=1), alpha * value_input_query))

        action_embeded = tf.matmul(tf.one_hot(action_input, num_actions), action_embedding)
        model_loss = tf.reduce_mean(tf.squared_difference(action_embeded + z_tar, z_pos))
        print("shape:", z_tar.shape, z_pos.shape, z_neg.shape, action_embeded.shape)
        # contrast_loss = tf.reduce_mean(tf.log(sum_negative) - positive)
        # print("shape2:", z.shape, negative.shape, positive.shape)
        # prediction_loss = tf.losses.mean_squared_error(value_input, v)
        total_loss = 0
        if "contrast" in loss_type:
            total_loss += contrast_loss
        if "regression" in loss_type:
            total_loss += beta * regression_loss
        elif "linear_model" in loss_type:
            total_loss += theta * model_loss

        model_func_vars = U.scope_vars(U.absolute_scope_name("model_func"))
        if "linear_model" in loss_type:
            model_func_vars.append(action_embedding)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                total_loss,
                                                var_list=model_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(total_loss, var_list=model_func_vars)
        # Create callable functions
        # update_target_fn will be called periodically to copy Q network to target Q network
        z_var_summary = tf.summary.scalar("z_var", tf.reduce_mean(tf.math.reduce_std(z, axis=1)))
        negative_summary = tf.summary.scalar("negative_dist", tf.reduce_mean(emb_dist(z_tar, z_neg)))
        positive_summary = tf.summary.scalar("positive_dist", tf.reduce_mean(emb_dist(z_tar, z_pos)))
        contrast_loss_summary = tf.summary.scalar("contrast loss", tf.reduce_mean(contrast_loss))
        regression_loss_summary = tf.summary.scalar("regression loss", tf.reduce_mean(contrast_loss))
        model_loss_summary = tf.summary.scalar("model loss", tf.reduce_mean(contrast_loss))
        # prediction_loss_summary = tf.summary.scalar("prediction loss", tf.reduce_mean(prediction_loss))
        total_loss_summary = tf.summary.scalar("total loss", tf.reduce_mean(total_loss))

        summaries = [z_var_summary, total_loss_summary]

        if "contrast" in loss_type:
            summaries += [negative_summary, positive_summary, contrast_loss_summary]
        if "regression" in loss_type:
            summaries.append(regression_loss_summary)
        if "linear_model" in loss_type:
            summaries.append(model_loss_summary)
        summary = tf.summary.merge(summaries)
        outputs = [z_tar]
        if "contrast" in loss_type:
            outputs += [z_pos, z_neg]
        elif "linear_model" in loss_type:
            outputs += [z_pos]
        outputs += [total_loss, summary]
        train = U.function(
            inputs=inputs,
            outputs=outputs,
            updates=[optimize_expr]
        )

        eval = U.function(
            inputs=inputs,
            outputs=outputs,
            updates=[]
        )
        z_func = U.function(
            inputs=[obs_input_query],
            outputs=[z, h],
        )
        norm_func = U.function(
            inputs=[obs_input_query],
            outputs=[tf.norm(z_tar,axis=1)]
        )
        return z_func, train, eval, norm_func
