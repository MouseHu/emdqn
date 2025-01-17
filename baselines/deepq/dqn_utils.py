"""This file includes a collection of utility functions that are useful for
implementing DQN."""
import gym
import tensorflow as tf
import numpy as np
import random
import imageio
import os
import cv2


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def huber_loss(self, y_true, y_pred, max_grad=1.):
    """Calculates the huber loss.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    err = tf.abs(y_true - y_pred, name='abs')
    mg = tf.constant(max_grad, name='max_grad')
    lin = mg * (err - .5 * mg)
    quad = .5 * err * err
    return tf.where(err < mg, quad, lin)


def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()


class ConstantSchedule(object):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def compute_exponential_averages(variables, decay):
    """Given a list of tensorflow scalar variables
    create ops corresponding to their exponential
    averages
    Parameters
    ----------
    variables: [tf.Tensor]
        List of scalar tensors.
    Returns
    -------
    averages: [tf.Tensor]
        List of scalar tensors corresponding to averages
        of al the `variables` (in order)
    apply_op: tf.runnable
        Op to be run to update the averages with current value
        of variables.
    """
    averager = tf.train.ExponentialMovingAverage(decay=decay)
    apply_op = averager.apply(variables)
    return [averager.average(v) for v in variables], apply_op


def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients)


def initialize_interdependent_variables(session, vars_list, feed_dict):
    """Initialize a list of variables one at a time, which is useful if
    initialization of some variables depends on initialization of the others.
    """
    vars_left = vars_list
    while len(vars_left) > 0:
        new_vars_left = []
        for v in vars_left:
            try:
                # If using an older version of TensorFlow, uncomment the line
                # below and comment out the line after it.
                # session.run(tf.initialize_variables([v]), feed_dict)
                session.run(tf.variables_initializer([v]), feed_dict)
            except tf.errors.FailedPreconditionError:
                new_vars_left.append(v)
        if len(new_vars_left) >= len(vars_left):
            # This can happend if the variables all depend on each other, or more likely if there's
            # another variable outside of the list, that still needs to be initialized. This could be
            # detected here, but life's finite.
            raise Exception("Cycle in variable dependencies, or extenrnal precondition unsatisfied.")
        else:
            vars_left = new_vars_left


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)


class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    @staticmethod
    def is_in_array(element, array):
        for x in array:
            if np.all(element == x):
                return True
        return False

    @staticmethod
    def switch_first_half(obs, obs_next, batch_size):
        half_size = int(batch_size / 2)
        tmp = obs[:half_size, ...]
        obs[:half_size, ...] = obs_next[:half_size, ...]
        obs_next[:half_size, ...] = tmp
        return obs, obs_next

    def _encode_sample(self, idxes):
        negidxes = []
        for i in range(len(idxes)):
            exclude = [(idxes[i] - 1) % self.size, idxes[i], (idxes[i] + 1) % self.size]
            exclude_state = [self._encode_observation(idx) for idx in exclude]

            tmp = random.randint(0, self.num_in_buffer - 2)

            # print(tmp in exclude)
            # print(np.array(exclude_state).shape)
            # print(self._encode_observation(tmp).shape)
            # print(self._encode_observation(tmp) in exclude_state)
            while tmp in exclude or self.is_in_array(self._encode_observation(tmp), exclude_state):
                tmp = random.randint(0, self.num_in_buffer - 2)
            negidxes.append(tmp)

        obs_batch = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        obs_neg_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in negidxes], 0)
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)
        self.switch_first_half(obs_batch, next_obs_batch, len(idxes))

        # print(obs_batch.shape,next_obs_batch.shape,obs_neg_batch.shape)
        # for x in (zip(obs_batch,next_obs_batch,obs_neg_batch)):
        #     print(x)
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, obs_neg_batch

    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = []
        while len(idxes) < batch_size:
            idx = random.randint(0, self.num_in_buffer - 2)
            if idx not in idxes and not self.done[idx]:
                idxes.append(idx)
        # idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx = idx + 1  # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            # print("yes")
            return self.obs[end_idx - 1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [self.obs[start_idx % self.size] for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 2)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            if len(self.obs.shape) <= 2:
                return self.obs[start_idx:end_idx]
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        # print("In store_frame", frame.shape)
        if self.obs is None:
            if len(frame.shape) < 2:
                dtype = np.uint16
            else:
                dtype = np.uint8
            self.obs = np.empty([self.size] + list(frame.shape), dtype=dtype)
            self.action = np.empty([self.size], dtype=np.int32)
            self.reward = np.empty([self.size], dtype=np.float32)
            self.done = np.empty([self.size], dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done

    def add_batch(self, obs, actions, rewards, dones):
        """
        just add a batch into the buffer
        :param obs:
        :param actions:
        :param rewards:
        :param dones:
        :return:
        """
        # print(obs)
        for ob, action, r, done in zip(obs, actions, rewards, dones):
            if len(ob.shape) >= 4:
                ob = np.squeeze(ob, axis=0)
            # print(ob)
            idx = self.store_frame(ob)
            self.store_effect(idx, action, r, done)


# gray scale wrapper
import cv2
from gym import spaces

sc_width, sc_height = 220, 220


def _process_frame(frame):
    frame = frame[15:-15, 15:-15, :]
    # print ("frame_shape", np.shape(frame))
    img_h, img_w, img_c = frame.shape
    img = np.reshape(frame, [img_h, img_w, img_c]).astype(np.float32)
    # print ("img_shape", np.shape(img))
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (sc_width, sc_width), interpolation=cv2.INTER_LINEAR)
    x_t = np.reshape(resized_screen, [sc_width, sc_width, 1])
    return x_t.astype(np.uint8)


class ProcessFrame(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrame, self).__init__(env)
        img_h, img_w, _ = self.env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(sc_width, sc_width, 1))
        try:
            self.num_envs = env.num_envs
        except AttributeError:
            self.num_envs = 1

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame(obs), reward, done, info

    def reset(self):
        return _process_frame(self.env.reset())


class GIFRecorder(gym.Wrapper):
    def __init__(self, video_path, record_video=False, env=None):
        super(GIFRecorder, self).__init__(env)
        self.record = record_video
        self.video_path = video_path
        self.images = []
        self.video_number = 0
        self.returns = 0
        self.num_steps = 0
        if not os.path.isdir(video_path):
            os.makedirs(video_path)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.record:
            # print(obs.shape)
            self.images.append(np.swapaxes(obs, 0, 1))
            self.returns += reward * 0.99 ** (self.num_steps)
            self.num_steps += 1
            if done:
                imageio.mimsave(self.video_path + "/{}_{}.gif".format(self.video_number, self.returns), self.images)
                self.images = []
                self.returns = 0
                self.num_steps = 0
                self.video_number += 1
                self.record = False
        return obs, reward, done, info


class VideoRecorder(gym.Wrapper):
    def __init__(self, video_path, record_video=False, env=None):
        super(VideoRecorder, self).__init__(env)
        self.record = record_video
        self.video_path = video_path
        self.images = []
        self.video_number = 0
        self.returns = 0
        self.num_steps = 0
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = None
        if not os.path.isdir(video_path):
            os.mkdir(video_path)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.record:
            # print(obs.shape)
            self.images.append(np.swapaxes(obs, 0, 1))
            self.returns += reward * 0.99 ** (self.num_steps)
            self.num_steps += 1
            if done:
                # print(self.images[0].shape)
                height, width = self.images[0].shape[0:2]
                self.writer = cv2.VideoWriter(self.video_path + "/{}_{}.avi".format(self.video_number, self.returns),
                                              self.fourcc, 20.0, (height, width))
                for frame in self.images:
                    self.writer.write(np.tile(frame,(1,1,3)))
                self.writer.release()
                # cv2.destroyAllWindows()
                self.writer = None
                # imageio.mimsave(, self.images)
                self.images = []
                self.returns = 0
                self.num_steps = 0
                self.video_number += 1
                self.record = False
        return obs, reward, done, info


class NoOpWrapperMK(gym.Wrapper):
    def __init__(self, env):
        super(NoOpWrapperMK, self).__init__(env)
        self.action_space = spaces.Discrete(env.action_space.n - 1)


class RandomBuffer(object):
    def __init__(self, size, frame_history_len):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

    @staticmethod
    def is_in_array(element, array):
        for x in array:
            if np.all(element == x):
                return True
        return False

    @staticmethod
    def switch_first_half(obs, obs_next, batch_size):
        half_size = int(batch_size / 2)
        tmp = obs[:half_size, ...]
        obs[:half_size, ...] = obs_next[:half_size, ...]
        obs_next[:half_size, ...] = tmp
        return obs, obs_next

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    # def _encode_sample(self, idxes):
    #     obs_batch = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
    #     act_batch = self.action[idxes]
    #     rew_batch = self.reward[idxes]
    #     next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
    #     done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)
    #
    #     return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask
    def _encode_sample(self, idxes):
        negidxes = []
        for i in range(len(idxes)):
            exclude = [(idxes[i] - 1) % self.size, idxes[i], (idxes[i] + 1) % self.size]
            exclude_state = [self._encode_observation(idx) for idx in exclude]

            tmp = random.randint(0, self.num_in_buffer - 2)

            # print(tmp in exclude)
            # print(np.array(exclude_state).shape)
            # print(self._encode_observation(tmp).shape)
            # print(self._encode_observation(tmp) in exclude_state)
            while tmp in exclude or self.is_in_array(self._encode_observation(tmp), exclude_state):
                tmp = random.randint(0, self.num_in_buffer - 2)
            negidxes.append(tmp)

        obs_batch = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        obs_neg_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in negidxes], 0)
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)
        self.switch_first_half(obs_batch, next_obs_batch, len(idxes))

        # print(obs_batch.shape,next_obs_batch.shape,obs_neg_batch.shape)
        # for x in (zip(obs_batch,next_obs_batch,obs_neg_batch)):
        #     print(x)
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, obs_neg_batch

    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        if batch_size >= self.size - 1:
            idxes = np.arange(batch_size)
            np.random.shuffle(idxes)
            return self._encode_sample(idxes)
        # idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        idxes = []
        while len(idxes) < batch_size:
            idx = random.randint(0, self.num_in_buffer - 2)
            if idx not in idxes and not self.done[idx]:
                idxes.append(idx)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1 + self.size) % self.size)

    def _encode_observation(self, idx):
        end_idx = idx + 1  # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx - 1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            # black frame for test and same frame for training, doesn't make sense
            # frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            frames = [self.obs[start_idx % self.size] for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 2)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            if len(self.obs.shape) <= 2:
                return self.obs[start_idx:end_idx]
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action = np.empty([self.size], dtype=np.int32)
            self.reward = np.empty([self.size], dtype=np.float32)
            self.done = np.empty([self.size], dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done

    def add_batch(self, obs, actions, rewards, dones):
        """
        just add a batch into the buffer
        :param obs:
        :param actions:
        :param rewards:
        :param dones:
        :return:
        """
        # print(obs)
        for ob, action, r, done in zip(obs, actions, rewards, dones):
            if len(ob.shape) >= 4:
                ob = np.squeeze(ob, axis=0)
            # print(ob)
            idx = self.store_frame(ob)
            self.store_effect(idx, action, r, done)
