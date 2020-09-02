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


def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def build_act_mer(make_obs_ph, model_func, num_actions, scope="deepq", secondary_scope="model_func",
                  reuse=None):
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


def contrastive_loss_fc(emb_cur, emb_next, emb_neg, margin=1, c_type='margin', num_neg=1, batch_size=32, emb_dim=32):
    if c_type is None or c_type == 'origin':
        return tf.reduce_mean(
            tf.maximum(emb_dist(emb_cur, emb_next) - emb_dist(emb_cur, emb_neg) + margin, 0)), []
    elif c_type == 'sqmargin':
        return tf.reduce_mean(emb_dist(emb_cur, emb_next) +
                              tf.maximum(0.,
                                         margin - emb_dist(emb_cur, emb_neg))), []

    elif c_type == 'margin':
        return tf.reduce_mean(emb_dist(emb_cur, emb_next) + tf.square(tf.maximum(0., margin -
                                                                                 tf.math.sqrt(
                                                                                     emb_dist(emb_cur,
                                                                                              emb_neg))))), []
    elif c_type == 'infonce':
        tau = 100
        emb_neg = tf.reshape(emb_neg, [batch_size, num_neg, emb_dim])
        emb_cur = tf.reshape(emb_cur, [batch_size, emb_dim, 1])
        emb_next = tf.reshape(emb_next, [batch_size, 1, emb_dim])
        negative = tf.matmul(emb_neg, emb_cur) / tau
        sum_negative = tf.squeeze(tf.reduce_mean(tf.exp(negative), axis=1))
        positive = tf.squeeze(tf.matmul(emb_next, emb_cur) / tau)
        contrast_loss = tf.reduce_mean(tf.log(sum_negative) - positive)
        negative_summary = tf.summary.scalar("negative dot", tf.reduce_mean(negative))
        positive_summary = tf.summary.scalar("positive dot", tf.reduce_mean(positive))
        sum_negative_summary = tf.summary.scalar("sum negative", tf.reduce_mean(sum_negative))
        norm_summary = tf.summary.scalar("contrast norm", tf.reduce_mean(tf.norm(emb_cur, axis=1)))
        total_summary = [negative_summary, positive_summary, sum_negative_summary, norm_summary]
        return contrast_loss, total_summary
    else:
        raise NotImplementedError


def acos_safe(x, eps=1e-4):
    slope = np.arccos(1 - eps) / eps
    # TODO: stop doing this allocation once sparse gradients with NaNs (like in
    # th.where) are handled differently.

    sign = tf.sign(x)
    out = tf.where(abs(x) <= 1 - eps, tf.acos(x), tf.acos(sign * (1 - eps)) - slope * sign * (abs(x) - 1 + eps))
    return out


def z_to_h(z, noise,latent_dim=32, batch_size=32):
    h_mean = z[..., :latent_dim]
    h_logvar = z[..., latent_dim:]
    # noise = tf.random.normal(shape=(batch_size, latent_dim))
    h = h_mean + noise * tf.exp(0.5 * h_logvar)
    return h


def build_train_mer_bvae_attention(input_type, obs_shape, encoder_func, decoder_func, num_actions, optimizer,
                                   grad_norm_clipping=None,
                                   gamma=1.0,
                                   scope="mfec", num_neg=10,
                                   latent_dim=32, alpha=0.1, beta=1e1, theta=10, loss_type=["contrast"], knn=4,
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
        obs_input_query = U.ensure_tf_input(input_type(obs_shape, None, name="obs_query"))
        obs_input_positive = U.ensure_tf_input(input_type(obs_shape, batch_size, name="enc_obs_pos"))
        obs_input_negative = U.ensure_tf_input(input_type(obs_shape, batch_size * num_neg, name="enc_obs_neg"))
        obs_input_neighbour = U.ensure_tf_input(input_type(obs_shape, batch_size * knn, name="enc_obs_neighbour"))

        obs_input_uniformity_u = U.ensure_tf_input(input_type(obs_shape, batch_size, name="enc_obs_uni_u"))
        obs_input_uniformity_v = U.ensure_tf_input(input_type(obs_shape, batch_size, name="enc_obs_uni_v"))

        obs_input_weighted_product_u = U.ensure_tf_input(input_type(obs_shape, batch_size, name="enc_obs_wp_u"))
        obs_input_weighted_product_v = U.ensure_tf_input(input_type(obs_shape, batch_size, name="enc_obs_wp_v"))

        value_input_weighted_product_u = tf.placeholder(tf.float32, [batch_size], name="value_u")
        value_input_weighted_product_v = tf.placeholder(tf.float32, [batch_size], name="value_v")

        value_input_query = tf.placeholder(tf.float32, [batch_size], name="value")
        value_input_neighbour = tf.placeholder(tf.float32, [batch_size, knn], name="neighbour_value")
        action_embedding = tf.Variable(tf.random_normal([num_actions, latent_dim * 2], stddev=1),
                                       name="action_embedding")
        action_input = tf.placeholder(tf.int32, [batch_size], name="action")
        action_input_causal = tf.placeholder(tf.int32, [batch_size], name="action")
        reward_input_causal = tf.placeholder(tf.float32, [batch_size], name="action")

        noise_input = tf.placeholder(tf.float32, [9, None, latent_dim])
        z_input = tf.placeholder(tf.float32,[None,latent_dim*2])
        inputs = [noise_input, obs_input_query]
        if "contrast" in loss_type:
            inputs += [obs_input_positive, obs_input_negative]
        if "regression" in loss_type:
            inputs += [value_input_query]
        if "linear_model" in loss_type:
            inputs += [action_input]
            if "contrast" not in loss_type:
                inputs += [obs_input_positive]
        if "fit" in loss_type:
            # if "contrast" not in loss_type:
            #     inputs+=[]
            inputs += [obs_input_neighbour, value_input_neighbour]
            if "regression" not in loss_type:
                inputs += [value_input_query]
        if "weight_product" in loss_type:
            inputs += [obs_input_uniformity_u, obs_input_uniformity_v, obs_input_weighted_product_u,
                       obs_input_weighted_product_v, value_input_weighted_product_u, value_input_weighted_product_v]
        if "causality" in loss_type:
            inputs += [reward_input_causal, action_input_causal]
        if "attention" in loss_type:
            inputs += [value_input_query]
        print("noise shape", noise_input.shape)
        z_old, *_,unmasked_z_old = encoder_func(
            obs_input_query.get(), noise_input[0],
            scope="target_model_func",
            reuse=False)

        z, h, h_mean, h_logvar, attention, _ = encoder_func(
            obs_input_query.get(), noise_input[1],
            scope="model_func",
            reuse=tf.AUTO_REUSE)

        reconstruct, value_predict = decoder_func(
            h,
            scope="decoder_func",
            reuse=False
        )

        reconstruct_visualize, value_visualize = decoder_func(
            z_to_h(z_input,noise_input[1]),
            scope="decoder_func",
            reuse=tf.AUTO_REUSE
        )
        z_pos, *_ = encoder_func(
            obs_input_positive.get(), noise_input[2],
            scope="model_func", reuse=True)

        z_neg, *_ = encoder_func(
            obs_input_negative.get(), noise_input[3],
            scope="model_func", reuse=True)

        z_uni_u, *_ = encoder_func(
            obs_input_uniformity_u.get(), noise_input[4],
            scope="model_func", reuse=True)
        z_uni_v, *_ = encoder_func(
            obs_input_uniformity_v.get(), noise_input[5],
            scope="model_func", reuse=True)
        z_wp_u, *_ = encoder_func(
            obs_input_weighted_product_u.get(), noise_input[6],
            scope="model_func", reuse=True)
        z_wp_v, *_ = encoder_func(
            obs_input_weighted_product_v.get(), noise_input[7],
            scope="model_func", reuse=True)

        z_pos = tf.reshape(z_pos, [-1, latent_dim * 2])
        z_tar = tf.reshape(z, [-1, latent_dim * 2])
        if "contrast" in loss_type:
            z_neg = tf.reshape(z_neg, [-1, latent_dim * 2])
            contrast_loss, contrast_summary = contrastive_loss_fc(z_tar, z_pos, z_neg, c_type=c_loss_type,
                                                                  num_neg=num_neg,
                                                                  batch_size=batch_size, emb_dim=latent_dim * 2)
            symmetry_loss, symmetry_summary = contrastive_loss_fc(z_pos, z_tar, z_neg, c_type=c_loss_type,
                                                                  num_neg=num_neg,
                                                                  batch_size=batch_size, emb_dim=latent_dim * 2)
            contrast_loss += symmetry_loss

        z_neighbour, *_ = encoder_func(
            obs_input_neighbour.get(), tf.reshape(noise_input[4:8], (-1, latent_dim)),
            scope="model_func",
            reuse=True)

        # fit loss
        z_neighbour = tf.reshape(z_neighbour, [-1, knn, latent_dim * 2])
        square_dist = tf.square(tf.tile(tf.expand_dims(z_tar, 1), [1, knn, 1]) - z_neighbour)
        neighbour_dist = tf.reduce_sum(square_dist, axis=2)
        neighbour_coeff = tf.math.softmax(-neighbour_dist / b, axis=1)
        coeff_sum = tf.reduce_mean(tf.reduce_sum(neighbour_coeff, axis=1))
        value_input_neighbour_mean = tf.reduce_mean(value_input_neighbour)
        fit_value = tf.reduce_sum(tf.multiply(neighbour_coeff, value_input_neighbour), axis=1)
        fit_loss = tf.reduce_mean(tf.abs(fit_value - value_input_query))

        # causality loss
        reward_input_causal = tf.reshape(reward_input_causal, [1, -1])
        reward_tile = tf.tile(reward_input_causal, [batch_size, 1])
        # reward_mask = (reward_tile - tf.transpose(reward_tile)) ** 2
        reward_mask = 1 - tf.cast(tf.equal((reward_tile - tf.transpose(reward_tile)), tf.constant(0.)), tf.float32)
        action_input_causal = tf.reshape(action_input_causal, [1, -1])
        action_tile = tf.tile(action_input_causal, [batch_size, 1])
        action_mask = tf.cast(tf.equal((action_tile - tf.transpose(action_tile)), tf.constant(0)), tf.float32)
        total_mask = tf.multiply(reward_mask, action_mask)
        z_tile = tf.tile(tf.expand_dims(z_tar, 1), [1, batch_size, 1])
        z_diff = z_tile - tf.transpose(z_tile, perm=[1, 0, 2])
        distance = tf.reduce_sum(z_diff ** 2, axis=2)
        exp_distance = tf.exp(-distance)
        causal_find_rate = (tf.reduce_sum(total_mask)) / (batch_size ** 2 - batch_size)
        causal_loss = tf.reduce_sum(tf.multiply(exp_distance, total_mask))

        # regularization loss
        regularization_loss = -tf.maximum(1., tf.reduce_mean(U.huber_loss(z_tar, 0.01)))
        regression_loss = tf.reduce_mean(
            tf.squared_difference(tf.norm(z_tar, axis=1), alpha * value_input_query)) + regularization_loss

        # linear model loss
        action_embeded = tf.matmul(tf.one_hot(action_input, num_actions), action_embedding)
        model_loss = tf.reduce_mean(tf.squared_difference(action_embeded + z_tar, z_pos)) + 0.01 * regularization_loss

        # weighted product loss
        uniformity_loss = tf.reduce_sum(tf.exp(2 * tf.reduce_sum(tf.multiply(z_uni_u, z_uni_v), axis=1) - 2))
        value_weight = (value_input_weighted_product_u - value_input_weighted_product_v) ** 2
        # angle = acos_safe(tf.reduce_sum(tf.multiply(z_wp_u, z_wp_v), axis=1))
        angle = tf.reduce_sum(tf.multiply(z_wp_u, z_wp_v), axis=1)
        weighted_product = tf.multiply(value_weight, angle)
        wp_loss = tf.reduce_sum(weighted_product)

        # attention loss

        attention_flatten = tf.layers.flatten(attention)
        # minimize activated area
        print("attention shape", attention_flatten.shape)
        num_pixel = int(attention_flatten.shape[-1])
        encoder_loss_var = -reduce_std(attention_flatten, axis=1)
        # encoder_loss_mean = 1. / num_pixel * tf.maximum(tf.square(tf.norm(attention_flatten, ord=2, axis=1)), 0.2*num_pixel)
        encoder_loss_mean = tf.maximum(tf.reduce_mean(tf.square(attention_flatten), axis=1), 0.2)
        att_encoder_loss = encoder_loss_mean + 0.001 * encoder_loss_var
        print("encoder loss  shape", att_encoder_loss.shape)
        # be predictive w.r.t value
        att_decoder_loss = tf.reduce_mean(tf.square(value_predict - value_input_query), axis=1)
        attention_loss = tf.reduce_mean(att_decoder_loss + att_encoder_loss)

        # vae loss
        delta = int(reconstruct.shape[1] - obs_input_query.get().shape[1])
        assert delta % 2 == 0
        print("delta", delta)
        reconstruct = reconstruct[:, delta // 2:-delta // 2, delta // 2:-delta // 2, :]
        vae_encoder_loss = tf.keras.losses.binary_crossentropy(tf.layers.flatten(obs_input_query.get()),
                                                               tf.layers.flatten(reconstruct))
        vae_decoder_loss = -0.5 * tf.reduce_sum(1 + h_logvar - h_mean ** 2 - tf.exp(h_logvar), axis=1)
        vae_loss = vae_encoder_loss + beta * vae_decoder_loss

        total_loss = 0
        if "contrast" in loss_type:
            total_loss += contrast_loss
        if "regression" in loss_type:
            total_loss += beta * regression_loss
        if "linear_model" in loss_type:
            total_loss += theta * model_loss
        if "fit" in loss_type:
            total_loss += beta * fit_loss
        if "causality" in loss_type:
            total_loss += theta * causal_loss
        if "weight_product" in loss_type:
            total_loss += uniformity_loss
            total_loss += wp_loss
        if "attention" in loss_type:
            total_loss += attention_loss

        total_loss += vae_loss
        model_func_vars = U.scope_vars(U.absolute_scope_name("model_func"))
        model_func_vars_update = copy.copy(model_func_vars)
        if "linear_model" in loss_type:
            model_func_vars_update.append(action_embedding)
        model_func_vars_update.append(U.scope_vars(U.absolute_scope_name("decoder_func")))
        target_model_func_vars = U.scope_vars(U.absolute_scope_name("target_model_func"))

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
        z_var_summary = tf.summary.scalar("z_var", tf.reduce_mean(tf.math.reduce_std(z, axis=1)))
        if "contrast" in loss_type:
            z_neg = tf.reshape(z_neg, [batch_size, num_neg, latent_dim * 2])
            negative_summary = tf.summary.scalar("negative_dist", tf.reduce_mean(emb_dist(z_tar, z_neg[:, 0, :])))
        positive_summary = tf.summary.scalar("positive_dist", tf.reduce_mean(emb_dist(z_tar, z_pos)))
        if "contrast" in loss_type:
            contrast_loss_summary = tf.summary.scalar("contrast loss", tf.reduce_mean(contrast_loss))
        regularization_loss_summary = tf.summary.scalar("regularization loss", tf.reduce_mean(regularization_loss))
        regression_loss_summary = tf.summary.scalar("regression loss", tf.reduce_mean(regression_loss))
        model_loss_summary = tf.summary.scalar("model loss", tf.reduce_mean(model_loss))
        fit_loss_summary = tf.summary.scalar("fit loss", tf.reduce_mean(fit_loss))
        fit_value_summary = tf.summary.scalar("fit value", tf.reduce_mean(fit_value))
        neighbour_value_summary = tf.summary.scalar("neighbour value", value_input_neighbour_mean)
        coeff_summary = tf.summary.scalar("coeff sum", coeff_sum)
        square_dist_summary = tf.summary.scalar("square_dist", tf.reduce_mean(square_dist))
        z_neighbour_summary = tf.summary.scalar("z_neighbour_mean", tf.reduce_mean(z_neighbour))
        # fit_loss_summary = tf.summary.scalar("fit loss", tf.reduce_mean(fit_loss))
        # prediction_loss_summary = tf.summary.scalar("prediction loss", tf.reduce_mean(prediction_loss))
        causal_efficiency_summary = tf.summary.scalar("causal efficiency", causal_find_rate)
        causal_loss_summary = tf.summary.scalar("causal loss", causal_loss)
        # reward_mask_summary = tf.summary.scalar("reward mask summary", debug_reward_mask)
        # action_mask_summary = tf.summary.scalar("action mask summary", debug_action_mask)
        uniformity_loss_summary = tf.summary.scalar("uniform loss", uniformity_loss)
        wp_loss_summary = tf.summary.scalar("weighted product loss", wp_loss)

        att_encoder_loss_summary = tf.summary.scalar("attention encoder loss", tf.reduce_mean(att_encoder_loss))
        # attention_norm_summary = tf.summary.scalar("attention norm",
        #                                            tf.reduce_mean(tf.reduce_sum(attention_flatten, axis=1)))
        att_decoder_loss_summary = tf.summary.scalar("attention decoder loss", tf.reduce_mean(att_decoder_loss))
        attention_loss_summary = tf.summary.scalar("attention  loss", tf.reduce_mean(attention_loss))

        vae_encoder_loss_summary = tf.summary.scalar("vae encoder loss", tf.reduce_mean(vae_encoder_loss))
        vae_decoder_loss_summary = tf.summary.scalar("vae decoder loss", tf.reduce_mean(vae_decoder_loss))
        vae_loss_summary = tf.summary.scalar("vae loss", tf.reduce_mean(vae_loss))

        total_loss_summary = tf.summary.scalar("total loss", tf.reduce_mean(total_loss))

        summaries = [z_var_summary, total_loss_summary, regularization_loss_summary]

        if "contrast" in loss_type:
            summaries += [negative_summary, positive_summary, contrast_loss_summary]
            summaries += contrast_summary
        if "regression" in loss_type:
            summaries.append(regression_loss_summary)
        if "linear_model" in loss_type:
            summaries.append(model_loss_summary)
            if "contrast" not in loss_type:
                summaries.append(positive_summary)
        if "fit" in loss_type:
            summaries.append(fit_loss_summary)
            summaries.append(fit_value_summary)
            summaries.append(neighbour_value_summary)
            summaries.append(coeff_summary)
            summaries.append(square_dist_summary)
            summaries.append(z_neighbour_summary)
        if "causality" in loss_type:
            summaries.append(causal_efficiency_summary)
            summaries.append(causal_loss_summary)
            # summaries.append(reward_mask_summary)
            # summaries.append(action_mask_summary)
        if "weight_product" in loss_type:
            summaries.append(uniformity_loss_summary)
            summaries.append(wp_loss_summary)
        if "attention" in loss_type:
            summaries.append(att_encoder_loss_summary)
            summaries.append(att_decoder_loss_summary)
            # summaries.append(attention_norm_summary)
            summaries.append(attention_loss_summary)

        summaries.append(vae_encoder_loss_summary)
        summaries.append(vae_decoder_loss_summary)
        summaries.append(vae_loss_summary)

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
            inputs=[noise_input, obs_input_query],
            outputs=[z_old],
        )
        unmasked_z_func = U.function(
            inputs=[noise_input, obs_input_query],
            outputs=[unmasked_z_old],
        )

        norm_func = U.function(
            inputs=[noise_input, obs_input_query],
            outputs=[tf.norm(z_tar, axis=1)]
        )

        attention_func = U.function(
            inputs=[noise_input, obs_input_query],
            outputs=[attention]
        )

        value_func = U.function(
            inputs=[noise_input, obs_input_query],
            outputs=[value_predict]
        )

        reconstruct_func = U.function(
            inputs=[noise_input, z_input],
            outputs=[reconstruct_visualize, value_visualize]
        )
        update_target_func = U.function([], [], updates=[update_target_expr])
        return z_func, unmasked_z_func,train, eval, norm_func, attention_func, value_func, reconstruct_func, update_target_func
