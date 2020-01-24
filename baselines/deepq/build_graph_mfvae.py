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


def build_act_mfvae(make_obs_ph, q_func, z_noise, num_actions, scope="deepq", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        q, q_deterministic, v_mean, v_logvar, z_mean, z_logvar, recon_obs = q_func(observations_ph.get(), z_noise,
                                                                                   num_actions,
                                                                                   scope="q_func",
                                                                                   reuse=tf.AUTO_REUSE)

        act = U.function(inputs=[observations_ph, z_noise],
                         outputs=[z_mean, z_logvar])

        return act


def build_train_mfvae(make_obs_ph, q_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0, scope="mfec",
                   alpha=1.0, beta=1.0, theta=1.0, latent_dim=32, vae=True, reuse=None):
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
    act_noise = tf.placeholder(tf.float32, [None, latent_dim], name="act_noise")
    act_f = build_act_mfvae(make_obs_ph, q_func, act_noise, num_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders

        # EMDQN

        obs_vae_input = U.ensure_tf_input(make_obs_ph("obs_vae"))
        z_noise_vae = tf.placeholder(tf.float32, [None, latent_dim], name="z_noise_vae")
        inputs = [obs_vae_input, z_noise_vae]
        qec_input = tf.placeholder(tf.float32, [None], name='qec')
        ib_inputs = [obs_vae_input, z_noise_vae, qec_input]

        q_vae, q_deterministic_vae, v_mean_vae, v_logvar_vae, z_mean_vae, z_logvar_vae, recon_obs = q_func(
            obs_vae_input.get(),
            z_noise_vae, num_actions,
            scope="q_func",
            reuse=True)
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        encoder_loss = -1 + z_mean_vae ** 2 + tf.exp(z_logvar_vae) - z_logvar_vae
        decoder_loss = tf.keras.losses.binary_crossentropy(tf.reshape(recon_obs, [-1]), tf.reshape(
            tf.dtypes.cast(obs_vae_input._placeholder, tf.float32), [-1]))
        print("here", z_mean_vae.shape, z_logvar_vae.shape, encoder_loss.shape, decoder_loss.shape)
        vae_loss = tf.reduce_mean(beta * encoder_loss + theta * decoder_loss)

        ib_loss = (v_mean_vae - tf.stop_gradient(tf.expand_dims(qec_input, 1))) ** 2 / tf.exp(
            v_logvar_vae) + v_logvar_vae
        print("here2", v_mean_vae.shape, tf.expand_dims(qec_input, 1).shape, v_logvar_vae.shape, ib_loss.shape)
        total_ib_loss = tf.reduce_mean(alpha * ib_loss + beta * encoder_loss)
        if vae:
            total_ib_loss += tf.reduce_mean(theta * decoder_loss)
        if grad_norm_clipping is not None:
            optimize_expr_vae = U.minimize_and_clip(optimizer,
                                                    vae_loss,
                                                    var_list=q_func_vars,
                                                    clip_val=grad_norm_clipping)
            optimize_expr_ib = U.minimize_and_clip(optimizer,
                                                   total_ib_loss,
                                                   var_list=q_func_vars,
                                                   clip_val=grad_norm_clipping)
        else:
            optimize_expr_vae = optimizer.minimize(vae_loss, var_list=q_func_vars)
            optimize_expr_ib = optimizer.minimize(ib_loss, var_list=q_func_vars)
        # Create callable functions
        # EMDQN
        z_var_summary = tf.summary.scalar("z_var", tf.reduce_mean(tf.exp(z_logvar_vae)))
        encoder_loss_summary = tf.summary.scalar("encoder loss", tf.reduce_mean(encoder_loss))
        decoder_loss_summary = tf.summary.scalar("decoder loss", tf.reduce_mean(decoder_loss))
        vae_loss_summary = tf.summary.scalar("vae total loss", vae_loss)

        ib_loss_summary = tf.summary.scalar("ib loss", tf.reduce_mean(ib_loss))
        total_ib_loss_summary = tf.summary.scalar("total ib loss", tf.reduce_mean(total_ib_loss))

        vae_summaries = [vae_loss_summary, z_var_summary, encoder_loss_summary, decoder_loss_summary]

        if vae:
            ib_summaries = [z_var_summary, encoder_loss_summary, decoder_loss_summary, ib_loss_summary,
                            total_ib_loss_summary]
        else:
            ib_summaries = [z_var_summary, encoder_loss_summary, decoder_loss_summary, ib_loss_summary,
                            total_ib_loss_summary]

        vae_summary = tf.summary.merge(vae_summaries)
        ib_summary = tf.summary.merge(ib_summaries)

        vae_train = U.function(
            inputs=inputs,
            outputs=[vae_loss, vae_summary],
            updates=[optimize_expr_vae]
        )
        ib_train = U.function(
            inputs=inputs,
            outputs=[total_ib_loss, ib_summary],
            updates=[optimize_expr_vae]
        )

        return act_f, vae_train, ib_train
