"""
This is the example config file
larger lr
beta no bias
lower explr
not target beta
"""
import numpy as np

# More one-char representation will be added in order to support
# other objects.
# The following a=10 is an example although it does not work now
# as I have not included a '10' object yet.
a = 10

# This is the map array that represents the map
# You have to fill the array into a (m x n) matrix with all elements
# not None. A strange shape of the array may cause malfunction.
# Currently available object indices are # they can fill more than one element in the array.
# 0: nothing
# 1: wall
# 2: ladder
# 3: coin
# 4: spike
# 5: triangle -------source
# 6: square ------ source
# 7: coin -------- target
# 8: princess -------source
# 9: player # elements(possibly more than 1) filled will be selected randomly to place the player
# unsupported indices will work as 0: nothing

map_array = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 5, 1, 0, 0, 0, 6, 0, 1],
    [1, 9, 9, 9, 1, 9, 9, 9, 9, 9, 1],
    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1],
    [1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1],
    [1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1],
    [1, 9, 2, 9, 9, 9, 2, 9, 9, 4, 1],
    [1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1],
    [1, 2, 1, 0, 2, 0, 1, 0, 2, 0, 1],
    [1, 2, 1, 9, 2, 7, 1, 9, 2, 8, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# set to true -> win when touching the object
# 0, 1, 2, 3, 4, 9 are not possible
end_game = {
    8: True,
}

rewards = {
    "positive": 0,      # when collecting a coin
    "win": 1,          # endgame (win)
    "negative": 0,    # endgame (die)
    "tick": 0           # living
}


######### dqn only #########
# ensure correct import
import os
import sys
__file_path = os.path.abspath(__file__)
__dqn_dir = '/'.join(str.split(__file_path, '/')[:-2]) + '/'
sys.path.append(__dqn_dir)
__cur_dir = '/'.join(str.split(__file_path, '/')[:-1]) + '/'

from baselines.deepq.dqn_utils import PiecewiseSchedule

# load the random sampled obs

def seed_func():
    return np.random.randint(0, 1000)

num_timesteps = 3e7
learning_freq = 4
# training iterations to go
num_iter = num_timesteps / learning_freq

# piecewise learning rate
lr_multiplier = 1.0
learning_rate = PiecewiseSchedule([
    (0, 1e-4 * lr_multiplier),
    (num_iter / 10, 1e-4 * lr_multiplier),
    (num_iter / 2,  5e-5 * lr_multiplier),
], outside_value=5e-5 * lr_multiplier)

# piecewise exploration rate
exploration = PiecewiseSchedule([
    (0, 1.0),
    (num_iter / 2, 0.7),
    (num_iter * 3 / 4, 0.1),
    (num_iter * 7 / 8, 0.05),
], outside_value=0.05)

######### transfer only #########

ppo_config = {
    'seed': seed_func,  # will override game settings
    'num_timesteps': num_timesteps,
    'replay_buffer_size': 1000000,
    'batch_size': 32,
    'gamma': 0.99,
    'learning_starts': 50000,
    'learning_freq': learning_freq,
    'frame_history_len': 4,
    'target_update_freq': 10000,
    'grad_norm_clipping': 10,
    'learning_rate': learning_rate,
    'exploration': exploration,
    'lambda_c':0.,  #1e-5
#    'eval_obs_array': eval_obs_array,  # TODO: construct some eval_obs_array
    'room_q_interval': 1e5,  # q_vals will be evaluated every room_q_interval steps
    'epoch_size': 5e4,  # you decide any way
    'config_name': str.split(__file_path, '/')[-1].replace('.py', ''),  # the config file name
}


map_config = {
    'map_array': map_array,
    'rewards': rewards,
    'end_game': end_game,
    'init_score': 0,
    'init_lives': 1,  # please don't change, not going to work
    # configs for dqn
    'ppo_config': ppo_config,
    # work automatically only for aigym wrapped version
    'fps': 1000,
    'frame_skip': 1,
    'force_fps': True,  # set to true to make the game run as fast as possible
    'display_screen': True,
    'episode_length': 1200,
    'episode_end_sleep': 0.,  # sec
}