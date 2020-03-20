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


def build_act_dueling(make_obs_ph, q_func, model_func, num_actions, input_dim=84 * 84 * 4, hash_dim=32, use_rp=False,
                      scope="deepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        if use_rp:
            latten_obs = tf.reshape(observations_ph.get(), [-1, input_dim])
            rp = tf.random.normal([input_dim, hash_dim], 0, 1 / np.sqrt(hash_dim))
            obs_hash_output = tf.matmul(latten_obs, rp)

        else:
            obs_hash_output, _ = model_func(
                observations_ph.get(), num_actions,
                scope="hash_func",
                reuse=False)
        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        q_values = q_func(observations_ph.get(), num_actions, scope="q_func")
        deterministic_actions = tf.argmax(q_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))

        act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=[output_actions, obs_hash_output],
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        return act


def build_train_dueling(make_obs_ph, q_func, model_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0,
                        scope="deepq", input_dim=84 * 84 * 4, hash_dim=32, use_rp=False, imitate=False, reuse=None):
    """Creates the train function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
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
    act_f = build_act_dueling(make_obs_ph, q_func, model_func, num_actions, input_dim, hash_dim, use_rp, scope=scope,
                              reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")
        if imitate:
            imitate_act_t_ph = tf.placeholder(tf.float32, [None, num_actions], name="imitate_action")
        # EMDQN
        value_t_ph = tf.placeholder(tf.float32, [None], name='value_t')
        value_tp1_ph = tf.placeholder(tf.float32, [None], name='value_tp1')
        value_tp1_masked = (1.0 - done_mask_ph) * value_tp1_ph
        # q network evaluation
        q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # reuse parameters from act
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

        # compute RHS of bellman equation
        q_target = rew_t_ph + gamma * value_tp1_masked

        # compute the error (potentially clipped)
        td_error = q_target - (q_t_selected + value_t_ph)
        td_summary = tf.summary.scalar("td error", tf.reduce_mean(td_error))
        # EMDQN
        print(q_t.shape)
        if imitate:
            imitation_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=imitate_act_t_ph, logits=q_t),
                                       axis=1)
            print(imitation_loss.shape)
            errors = U.huber_loss(td_error) + imitation_loss
        else:
            errors = U.huber_loss(td_error)
        total_summary = tf.summary.scalar("total error", tf.reduce_mean(errors))

        value_summary = tf.summary.scalar("value_t", tf.reduce_mean(value_t_ph))
        value_tp1_summary = tf.summary.scalar("value_tp1", tf.reduce_mean(value_tp1_ph))
        q_summary = tf.summary.scalar("estimated qs", tf.reduce_mean(q_t_selected))
        summaries=[td_summary, total_summary, value_summary, value_tp1_summary, q_summary]
        if imitate:
            imitate_summary = tf.summary.scalar("imitate loss", tf.reduce_mean(imitation_loss))
            summaries.append(imitate_summary)
        summary = tf.summary.merge(summaries)

        weighted_error = tf.reduce_mean(importance_weights_ph * errors)
        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                weighted_error,
                                                var_list=q_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        inputs = [
            obs_t_input,
            act_t_ph,
            rew_t_ph,
            done_mask_ph,
            importance_weights_ph,
            value_t_ph,
            value_tp1_ph
        ]
        if imitate:
            inputs.append(imitate_act_t_ph)
        # Create callable functions
        # EMDQN
        train = U.function(
            inputs=inputs,
            outputs=[td_error, summary],
            updates=[optimize_expr]
        )

        return act_f, train
