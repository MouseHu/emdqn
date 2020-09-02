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
import copy


def acos_safe(x, eps=1e-4):
    slope = np.arccos(1 - eps) / eps
    # TODO: stop doing this allocation once sparse gradients with NaNs (like in
    # th.where) are handled differently.

    sign = tf.sign(x)
    out = tf.where(abs(x) <= 1 - eps, tf.acos(x), tf.acos(sign * (1 - eps)) - slope * sign * (abs(x) - 1 + eps))
    return out


def build_train_dbc(input_type, obs_shape, repr_func, model_func, num_actions, optimizer, grad_norm_clipping=None,
                    gamma=1.0,
                    scope="mfec", num_neg=10,
                    latent_dim=32, alpha=1, beta=1e2, theta=10, loss_type=["contrast"], knn=4,
                    c_loss_type="margin", b=100, batch_size=32,
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
    if c_loss_type != "infonce":
        assert num_neg == 1
    # z_func = build_act_contrast(make_obs_ph, model_func, num_actions, scope=scope, secondary_scope="model_func",
    #                             reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders

        # EMDQN
        # tau = tf.placeholder(tf.float32, [1], name='tau')
        # momentum = tf.placeholder(tf.float32, [1], name='momentum')

        # make_obs_ph = lambda name: input_type(obs_shape, batch_size, name=name),

        magic_num = tf.get_variable(name='magic', shape=[1])
        obs_input_u = U.ensure_tf_input(input_type(obs_shape, None, name="obs_u"))
        obs_input_u_tp1 = U.ensure_tf_input(input_type(obs_shape, None, name="obs_u_tp1"))
        obs_input_v = U.ensure_tf_input(input_type(obs_shape, None, name="obs_v"))

        action_input = tf.placeholder(tf.int32, [batch_size], name="action")
        reward_input = tf.placeholder(tf.float32, [batch_size], name="action")

        inputs = [obs_input_u, obs_input_u_tp1, obs_input_v, action_input, reward_input]
        z_old = repr_func(
            obs_input_u.get(), num_actions,
            scope="target_repr_func",
            reuse=False)

        z_u = repr_func(
            obs_input_u.get(), num_actions,
            scope="repr_func",
            reuse=tf.AUTO_REUSE)

        z_u_tp1 = repr_func(
            obs_input_u_tp1.get(), num_actions,
            scope="repr_func",
            reuse=tf.AUTO_REUSE)

        z_v = repr_func(
            obs_input_v.get(), num_actions,
            scope="repr_func",
            reuse=tf.AUTO_REUSE)

        z_u_tp1_predict, r_u_predict = model_func(
            z_u, num_actions,
            scope="model_func",
            reuse=tf.AUTO_REUSE)

        z_v_tp1_predict, r_v_predict = model_func(
            z_v, num_actions,
            scope="model_func",
            reuse=tf.AUTO_REUSE)

        # total_loss = 0
        # reprsentation loss
        dist_bisimulation = tf.reduce_max(tf.abs(r_u_predict - r_v_predict) + gamma * tf.reduce_sum(
            tf.square(z_u_tp1_predict - z_v_tp1_predict), axis=2), axis=1)
        dist_bisimulation = tf.stop_gradient(dist_bisimulation)
        repr_loss = tf.losses.mean_squared_error(tf.norm(z_u - z_v, ord=1, axis=1), dist_bisimulation)

        # model loss
        z_u_tp1_selected = tf.gather(z_u_tp1_predict, action_input, axis=1, batch_dims=0)
        r_u_selected = tf.gather(r_u_predict, action_input, axis=1, batch_dims=0)
        transition_loss = tf.losses.mean_squared_error(z_u_tp1, tf.stop_gradient(z_u_tp1_selected))
        reward_loss = tf.losses.mean_squared_error(reward_input, tf.stop_gradient(r_u_selected))
        model_loss = transition_loss + reward_loss

        total_loss = repr_loss + alpha * model_loss

        model_func_vars = U.scope_vars(U.absolute_scope_name("repr_func"))
        model_func_vars_update = copy.copy(model_func_vars) + U.scope_vars(U.absolute_scope_name("model_func"))

        target_model_func_vars = U.scope_vars(U.absolute_scope_name("repr_model_func"))

        update_target_expr = []
        for var in model_func_vars:
            print(var.name, var.shape)
        for var_target in target_model_func_vars:
            print(var_target.name, var_target.shape)

        for var, var_target in zip(sorted(model_func_vars, key=lambda v: v.name),
                                   sorted(target_model_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                total_loss,
                                                var_list=model_func_vars_update,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(total_loss, var_list=model_func_vars_update)
        # Create callable functions
        # update_target_fn will be called periodically to copy Q network to target Q network
        z_var_summary = tf.summary.scalar("z_var", tf.reduce_mean(tf.math.reduce_std(z_u, axis=1)))
        total_loss_summary = tf.summary.scalar("total loss", tf.reduce_mean(total_loss))
        transition_loss_summary = tf.summary.scalar("transition loss", tf.reduce_mean(transition_loss))
        reward_loss_summary = tf.summary.scalar("reward loss", tf.reduce_mean(reward_loss))
        model_loss_summary = tf.summary.scalar("model loss", tf.reduce_mean(model_loss))
        repr_loss_summary = tf.summary.scalar("repr loss", tf.reduce_mean(repr_loss))

        summaries = [z_var_summary, total_loss_summary, transition_loss_summary, reward_loss_summary,
                     model_loss_summary, repr_loss_summary]

        summary = tf.summary.merge(summaries)
        outputs = [total_loss, summary]
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
            inputs=[obs_input_u],
            outputs=[z_old],
        )
        update_target_func = U.function([], [], updates=[update_target_expr])
        return z_func, train, eval, update_target_func
