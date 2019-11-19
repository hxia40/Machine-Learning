"""
Solving environment using Policy Itertion.
Author : Wei Feng
"""
import numpy as np
import gym
from gym import wrappers


class PI:
    def __init__(self, env):
        self.env = env

    def policy_evaluation(self, policy, gamma):
        V = np.zeros(self.env.nS)
        THETA = 1e-10
        delta = float("inf")

        while delta > THETA:
            delta = 0
            for s in range(self.env.nS):
                expected_value = 0
                for action, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in self.env.P[s][action]:
                        expected_value += action_prob * prob * (reward + gamma * V[next_state])
                delta = max(delta, np.abs(V[s] - expected_value))
                # print "delta:", delta
                V[s] = expected_value

        return V

    def next_best_action(self, s, V, gamma):
        action_values = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in self.env.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state])
        return np.argmax(action_values), np.max(action_values)

    def optimize(self, gamma = 1):
        policy = np.tile(np.eye(self.env.nA)[1], (self.env.nS, 1))

        is_stable = False

        round_num = 0

        while not is_stable:
            is_stable = True
            print("\nRound Number:" + str(round_num))
            round_num += 1
            # print("Current Policy")
            # print(
            #     np.reshape([env.get_action_name(entry) for entry in [np.argmax(policy[s]) for s in range(self.env.nS)]],
            #                (8,8)))

            V = self.policy_evaluation(policy, gamma)
            # print("Expected Value accoridng to Policy Evaluation")
            # print(np.reshape(V, (8, 8)))

            for s in range(self.env.nS):
                action_by_policy = np.argmax(policy[s])
                best_action, best_action_value = self.next_best_action(s, V, gamma)
                # print("\nstate=" + str(s) + " action=" + str(best_action))
                policy[s] = np.eye(self.env.nA)[best_action]
                if action_by_policy != best_action:
                    is_stable = False

        policy = [np.argmax(policy[s]) for s in range(self.env.nS)]
        return policy


if __name__ == '__main__':
    env_name  = 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    pi = Agent(env)
    optimal_policy = pi.optimize(gamma=1)
    policy_score = evaluate_policy(env, optimal_policy, n=1000)
    print('Policy average score = ', policy_score)