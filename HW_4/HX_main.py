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
from HX_QLearner import QLearningTable, QLearningTableNC
from HX_XQL import XQLearningTable
from HX_XQPlusL import XQPlusLearningTable
from HX_NChain import NChainEnv
from HX_maze import generate_random_map, FrozenLakeEnv
import gym
import time
import math
import random


def Q_FL0(learning_rate = 0.01):
    Q_file = open('QLearner.txt', 'a')

    episode_list = []
    reward_list = []
    reward_list_jr = []
    time_list = []
    time_list_jr = []

    range_end = 2000000
    for episode in range(range_end):
        # alpha = (1 - math.log(episode+1, 10) / math.log(range_end, 10))/10
        alpha = learning_rate
        if (episode + 1) % (range_end / 100) == 0:
            print "episode = ", episode + 1, "learnng rate = ", alpha, "reward = ", np.mean(reward_list_jr)
            episode_list.append(episode + 1)
            time_list.append(np.mean(time_list_jr))
            time_list_jr = []
            reward_list.append(np.mean(reward_list_jr))

            reward_list_jr = []
        # initial observation
        observation = env_FL0.reset()
        start_time = time.time()
        while True:
            # # fresh env
            # env_FL0.render()

            # QL choose action based on observation
            action = QL_FL0.choose_action(str(observation))
            # print env_FL0.step(action)

            # QL take action and get next observation and reward
            observation_, reward, done, info = env_FL0.step(action)

            # QL learn from this transition
            QL_FL0.learn(str(observation), action, reward, str(observation_), alpha)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                time_list_jr.append(time.time()-start_time)
                reward_list_jr.append(reward)
                break

    # Q_file.write('episodes:')
    # Q_file.write(str(episode_list))
    # Q_file.write('\n')
    # Q_file.write('rewards:')
    Q_file.write(str(reward_list))
    Q_file.write('\n')
    # Q_file.write('time_consumption:')
    # Q_file.write(str(time_list))
    Q_file.close()

    # end of game
    print('game over')
    # env.destroy()


def Q_NC(random_seed=1):
    Q_file = open('QLearnerNC.txt', 'w')
    episode_list = []
    reward_list = []
    reward_list_jr = []
    time_list = []
    time_list_jr = []

    range_end = 10000
    env_NC.seed(random_seed)
    for episode in range(range_end):
        # alpha = (1 - math.log(episode+1, 10) / math.log(range_end, 10))/10
        alpha = 0.01
        if (episode + 1) % (range_end / 100) == 0:
            print "episode = ", episode + 1, "learnng rate = ", alpha, "reward = ", np.mean(reward_list_jr)
            episode_list.append(episode + 1)
            time_list.append(np.mean(time_list_jr))
            time_list_jr = []
            reward_list.append(np.mean(reward_list_jr))

            reward_list_jr = []
        # initial observation
        observation = env_NC.reset()
        start_time = time.time()
        while True:
            # # fresh env
            # env_NC.render()

            # QL choose action based on observation
            action = QL_NC.choose_action(observation)
            # print env_NC.step(action)

            # QL take action and get next observation and reward
            observation_, reward, done, info = env_NC.step(action)
            # print action
            # print env_NC.step(action)

            # QL learn from this transition
            QL_NC.learn(observation, action, reward, observation, alpha)

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

    random.seed(1)

    env_FL0 = FrozenLakeEnv(desc=generate_random_map(size=32, p=0.9), map_name=None,is_slippery=False)
    env_FL0.render()

    '''FrozenLake - XQ-plus learning'''

    # print "XQPlusLearningTable"
    # for i in range(1):
    #     QL_FL0 = XQPlusLearningTable(actions=list(range(env_FL0.nA)),
    #                             # learning_rate=0.1,
    #                             reward_decay=0.99,
    #                             e_greedy=0.9,
    #                             exploration_decay=0.99,
    #                             verbose = 1)
    #     Q_FL0(learning_rate = 0.01)

    '''FrozenLake - XQ-learning'''
    #
    print "XQLearningTable"
    for i in range(1):
        QL_FL0 = XQLearningTable(actions=list(range(env_FL0.nA)),
                                # learning_rate=0.1,
                                reward_decay=0.99,
                                e_greedy=0.9,
                                verbose = 1)
        Q_FL0(learning_rate = 0.01)

    '''FrozenLake - Q-learning'''

    # print "QLearningTable"
    # for i in range(10):
    #     QL_FL0 = QLearningTable(actions=list(range(env_FL0.nA)),
    #                             # learning_rate=0.1,
    #                             reward_decay=0.99,
    #                             e_greedy=0.9,
    #                             verbose = True)
    #     Q_FL0(learning_rate = 0.01)

    '''NChain'''
    # env_NC = NChainEnv(gym.Env)
    #
    # # setting of the NC environemnt. n=5, slip=0.2, small=2, large=10
    # env_NC.n = 1000
    # env_NC.slip = 0
    # env_NC.small = 2
    # env_NC.large = 1000000
    #
    # # print env_NC.n, env_NC.slip, env_NC.small, env_NC.large
    #
    # QL_NC = QLearningTableNC(actions=list(range(2)),
    #                          learning_rate=0.01,
    #                          reward_decay=0.999,
    #                          e_greedy=0.999,
    #                          total_length=env_NC.n)
    # Q_NC(random_seed=1)
    # #
