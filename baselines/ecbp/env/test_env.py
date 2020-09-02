import gym
from pyvirtualdisplay import Display
import time

def main():
    keyboard_config = {"w": 3, "s": 4, "d": 1, "a": 0, "x": 2, "p": 5, "3": 3, "4": 4, "1": 1, "0": 0, "2": 2, "5": 5}
    display = Display(visible=1, size=(1080, 720))
    display.start()

    env = gym.make('MountainCar-v0')
    env.reset()
    returns = 0
    while True:
        # print("input action:",end=" ")
        # char = readchar.readkey()
        # # char = input("input action:")
        # action = keyboard_config.get(char, 6)
        # if action == 6:
        #     exit(0)
        #
        # print(" ")

        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print("obs",obs)
        time.sleep(0.1)
        returns+=reward
        env.render()
        if done:
            print("ended.",returns)
            returns = 0
            env.reset()


if __name__ == '__main__':
    main()
