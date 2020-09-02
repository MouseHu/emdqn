import gym
from pyvirtualdisplay import Display
from gym.envs.registration import register

from mujoco import *
import readchar


def main():
    keyboard_config = {"w": 3, "s": 4, "d": 1, "a": 0, "x": 2, "p": 5, "3": 3, "4": 4, "1": 1, "0": 0, "2": 2, "5": 5}
    display = Display(visible=1, size=(1080, 720))
    display.start()

    goal_args = [[0.0, 16.0], [0+1e-3, 16 + 1e-3],[0.0, 6.0], [15+1e-3, 10 + 1e-3]]
    random_start = False
    # The episode length for test is 500
    max_timestep = 500
    register(
        id='PointMazeTest-v10',
        entry_point='mujoco.create_maze_env:create_maze_env',
        kwargs={'env_name': 'DiscretePointBlock', 'goal_args': goal_args, 'maze_size_scaling': 4, 'random_start': random_start},
        max_episode_steps=max_timestep,
    )
    env = gym.make('PointMazeTest-v10')
    env.reset()
    while True:
        print("input action:",end=" ")
        char = readchar.readkey()
        # char = input("input action:")
        action = keyboard_config.get(char, 6)
        if action == 6:
            exit(0)

        print(" ")
        # action = env.unwrapped.pseudo_action_space.sample()
        obs, reward, done, info = env.step(action)
        print("reward",reward)
        print("obs",obs)
        env.render()
        if done:
            env.reset()


if __name__ == '__main__':
    main()
