import gym


def main():
    env = gym.make('toy_envs:Point-v0')

    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            env.reset()


if __name__ == '__main__':
    main()
