"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time


class XQLearningTable:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, verbose=False):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.x_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.new_state_counter = 0
        self.verbose = verbose

    def choose_action(self, observation):
        self.check_state_exist(observation)
        state_action = self.q_table.loc[observation, :]
        state_exploration = self.x_table.loc[observation, :]
        # action selection
        # if there's some action have not been tried (i.e. value =1 in the x table), force the agent to try it first.
        if not all(state_exploration == 0):
            if self.verbose >=2 :
                print("lets see what the state_explloration is!!!!\n", state_exploration)
            action = np.random.choice(state_exploration[state_exploration == 1].index)
            if self.verbose >=2 :
                print("action chosen:", action, '\n')
        else:
            if np.random.uniform() < self.epsilon:

                # choose best action -  some actions may have the same value, randomly choose on in these actions
                # state_action = self.q_table.loc[observation, :]
                action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            else:
                # choose random action
                action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, alpha):
        # update q table and x table
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        # self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update , Morvan's original
        self.q_table.loc[s, a] += alpha * (q_target - q_predict)  # update q table , HX self defined
        self.x_table.loc[s, a] = 0  # update x table , HX self defined
        if self.verbose >= 2:
            print('\n Q table is:\n', self.q_table)
            print('\n X table is:\n', self.x_table)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.new_state_counter += 1
            if self.verbose >= 1:
                print('========adding', self.new_state_counter,'th new state====== : ', state)
            if self.verbose >= 2:
                print('\n Q table is:\n', self.q_table)
                print('\n X table is:\n', self.x_table)
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
            # append new state to x table
            self.x_table = self.x_table.append(
                pd.Series(
                    [1] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


class XQLearningTableNC:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, total_length = 10, verbose=False):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.x_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.new_state_counter = 0
        self.verbose = verbose
        self.total_length = total_length

    def choose_action(self, observation):
        self.check_state_exist(observation)
        state_action = self.q_table.loc[observation, :]
        state_exploration = self.x_table.loc[observation, :]
        # action selection
        # if there's some action have not been tried (i.e. value =1 in the x table), force the agent to try it first.
        if not all(state_exploration == 0):
            if self.verbose >= 2:
                print("lets see what the state_exploration is!!!!\n", state_exploration)
            action = np.random.choice(state_exploration[state_exploration == np.max(state_exploration)].index)
            if self.verbose >= 2:
                print("action chosen:", action, '\n')
        else:
            if np.random.uniform() < self.epsilon:

                # choose best action -  some actions may have the same value, randomly choose on in these actions
                # state_action = self.q_table.loc[observation, :]
                action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            else:
                # choose random action
                action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, alpha):
        # update q table and x table
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != self.total_length:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        # self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update , Morvan's original
        self.q_table.loc[s, a] += alpha * (q_target - q_predict)  # update q table , HX self defined
        self.x_table.loc[s, a] = max(self.x_table.loc[s, a] -1 , 0)  # update x table , HX self defined
        if self.verbose >= 2:
            print('\n Q table is:\n', self.q_table)
            print('\n X table is:\n', self.x_table)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.new_state_counter += 1
            if self.verbose >= 1:
                print('========adding', self.new_state_counter,'th new state====== : ', state)
            if self.verbose >= 2:
                print('\n Q table is:\n', self.q_table)
                print('\n X table is:\n', self.x_table)
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
            # append new state to x table
            self.x_table = self.x_table.append(
                pd.Series(
                    [1] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )