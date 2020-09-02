import gym
from pyvirtualdisplay import Display
from gym.envs.registration import register

def main():
    display = Display(visible=1, size=(1080, 720))
    display.start()
    register(
        id='Point-v0',
        entry_point='toy_envs.envs:PointEnv',
        max_episode_steps=200
    )
    env = gym.make('Point-v0')

    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            env.reset()


if __name__ == '__main__':
    main()
