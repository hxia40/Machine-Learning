"""
Solving environment using Value Itertion.
Author : Wei Feng
"""
import numpy as np
import gym
from gym import wrappers
def run_episode(env, policy, gamma = 1.0, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()  # Resetting the environment will return an integer. This number will be our initial state.
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        # total_reward += (gamma ** step_idx * reward)
        # the above code is from Moustafa Alzantot , which this is problematic.
        # As the policy's target here is never to finish in shortest time. Rather,
        # the only thing matters is that if u can successfully recover ur stuff, or drop into one of the ice-hole.
        total_reward += reward    # HX
        step_idx += 1
        if done:
            break
    # print "total_reward:", total_reward
    return total_reward

def evaluate_policy(env, policy, gamma = 1.0,  n = 1000):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode(env, policy, gamma = gamma, render = False)
            for _ in range(n)]
    return np.mean(scores)

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

if __name__ == '__main__':
    env_name  = 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    pi = VI(env)
    optimal_policy = pi.optimize(gamma=1)
    policy_score = evaluate_policy(env, optimal_policy, n=1000)
    print('Policy average score = ', policy_score)

