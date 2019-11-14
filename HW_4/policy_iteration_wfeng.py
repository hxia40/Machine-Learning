"""
Solving environment using Policy Itertion.
Author : Wei Feng
"""
import numpy as np
import gym
from gym import wrappers


class Agent:
    def __init__(self, env):
        self.env = env

    def policy_evaluation(self, policy):
        V = np.zeros(self.env.nS)
        THETA = 0.0001
        delta = float("inf")

        while delta > THETA:
            delta = 0
            for s in range(self.env.nS):
                expected_value = 0
                for action, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in self.env.P[s][action]:
                        expected_value += action_prob * prob * (reward + DISCOUNT_FACTOR * V[next_state])
                delta = max(delta, np.abs(V[s] - expected_value))
                V[s] = expected_value

        return V

    def next_best_action(self, s, V):
        action_values = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in self.env.P[s][a]:
                action_values[a] += prob * (reward + DISCOUNT_FACTOR * V[next_state])
        return np.argmax(action_values), np.max(action_values)

    def optimize(self):
        policy = np.tile(np.eye(self.env.nA)[1], (self.env.nS, 1))

        is_stable = False

        round_num = 0

        while not is_stable:
            is_stable = True

            print("\nRound Number:" + str(round_num))
            round_num += 1

            print("Current Policy")
            print(
                np.reshape([env.get_action_name(entry) for entry in [np.argmax(policy[s]) for s in range(self.env.nS)]],
                           self.env.shape))

            V = self.policy_evaluation(policy)
            print("Expected Value accoridng to Policy Evaluation")
            print(np.reshape(V, self.env.shape))

            for s in range(self.env.nS):
                action_by_policy = np.argmax(policy[s])
                best_action, best_action_value = self.next_best_action(s, V)
                # print("\nstate=" + str(s) + " action=" + str(best_action))
                policy[s] = np.eye(self.env.nA)[best_action]
                if action_by_policy != best_action:
                    is_stable = False

        policy = [np.argmax(policy[s]) for s in range(self.env.nS)]
        return policy


# if __name__ == '__main__':
#     env_name  = 'FrozenLake8x8-v0'
#     env = gym.make(env_name)
#     optimal_policy = policy_iteration(env, gamma = 1.0)
#     scores = evaluate_policy(env, optimal_policy, gamma = 1.0)
#     print('Average scores = ', np.mean(scores))