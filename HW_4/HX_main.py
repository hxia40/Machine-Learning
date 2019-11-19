"""
Reinforcement learning


Reference:
Morzan,
wfeng,
Moustafa Alzantot (malzantot@ucla.edu)

"""
import numpy as np
from HX_policy_iteration import PI
from HX_value_iteration import VI
from HX_QLearner import QLearningTable
import gym
import time
import math


def Q_update():
    Q_file = open('QLearner.txt', 'w')
    episode_list = []
    reward_list = []
    reward_list_jr = []
    time_list = []
    time_list_jr = []

    range_end = 1000000
    for episode in range(range_end):
        # alpha = 1 - math.log(episode+1, 10) / math.log(range_end, 10)
        alpha = 0.0001
        if (episode + 1) % (range_end / 100) == 0:
            print "episode = ", episode + 1, "learnng rate = ", alpha
            episode_list.append(episode + 1)
            time_list.append(np.mean(time_list_jr))
            time_list_jr = []
            reward_list.append(np.mean(reward_list_jr))
            reward_list_jr = []
        # initial observation
        observation = env.reset()
        start_time = time.time()
        while True:
            # # fresh env
            # env.render()

            # QL choose action based on observation
            action = QL.choose_action(str(observation))
            # print env.step(action)

            # QL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)

            # QL learn from this transition
            QL.learn(str(observation), action, reward, str(observation_), alpha)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                time_list_jr.append(time.time()-start_time)
                reward_list_jr.append(reward)
                break

    Q_file.write('episodes:')
    Q_file.write(str(episode_list))
    Q_file.write('\n')
    Q_file.write('rewards:')
    Q_file.write(str(reward_list))
    Q_file.write('\n')
    Q_file.write('time_consumption:')
    Q_file.write(str(time_list))

    # end of game
    print('game over')
    # env.destroy()

if __name__ == "__main__":
    # env_name  = 'FrozenLake8x8-v0'
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)

    QL = QLearningTable(actions=list(range(env.nA)), learning_rate=1, reward_decay=0.9, e_greedy=0.9)
    Q_update()
