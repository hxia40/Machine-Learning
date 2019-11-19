"""
Solving environment using Value Itertion.
Author : Wei Feng
"""
import numpy as np
import gym
from gym import wrappers


class VI:
    def __init__(self, env):
        self.env = env
        self.V = np.zeros(env.nS)

    def next_best_action(self, s, V, gamma=1):
        action_values = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state])
        return np.argmax(action_values), np.max(action_values)

    def optimize(self, gamma =1):
        THETA = 1e-2000
        delta = float("inf")
        round_num = 0

        while delta > THETA:
            delta = 0
            print("\nValue Iteration: Round " + str(round_num))
            print(np.reshape(self.V,(8,8)))
            # print(np.reshape(self.V, env.shape))
            for s in range(env.nS):
                best_action, best_action_value = self.next_best_action(s, self.V, gamma)
                delta = max(delta, np.abs(best_action_value - self.V[s]))
                self.V[s] = best_action_value
            round_num += 1

        policy = np.zeros(env.nS)
        for s in range(env.nS):
            best_action, best_action_value = self.next_best_action(s, self.V, gamma)
            policy[s] = best_action

        return policy

