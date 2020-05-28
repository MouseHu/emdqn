import parser
import numpy as np
from baselines.ecbp.util import *


def init_experiment():
    subdir = (datetime.datetime.now()).strftime("%m-%d-%Y-%H:%M:%S") + " " + args.comment
    tf_writer = tf.summary.FileWriter(os.path.join(args.log_dir, subdir), tf.get_default_graph())
    make_logger("ecbp", os.path.join(args.log_dir, subdir, "logger.log"))

    value_summary = tf.Summary()
    value_summary.value.add(tag='discount_reward_mean')
    value_summary.value.add(tag='non_discount_reward_mean')
    value_summary.value.add(tag='steps')
    value_summary.value.add(tag='episodes')

    return tf_writer, value_summary


def log(info, eval):
    for name, value in info.items():
        if eval:
            name = name + "_eval"
        logger.record_tabular(name, value)


def run_one_episode(eval):
    obs = env.reset()
    while True:
        action = agent.act(np.array(obs)[None], is_train=not eval)
        obs, rew, done, info = env.step(action, is_train=not eval)
        agent.observe(action, rew, obs, done, train=not eval)

        if done:
            break
    return env.get_info() + agent.get_info()


def run_experiment():
    num_episodes, num_steps = 0, 0
    while True:
        eval = 1
        info = run_one_episode(eval)
        num_steps += info["steps"]
        log(info, eval)
        if num_steps > args.num_steps:
            agent.finish()
            break


if __name__ == "__main__":
    args = parse_args()
    tf_writer, summary = init_experiment()
    env = create_env(args)
    agent = make_agent(args, env, tf_writer)
    run_experiment()
