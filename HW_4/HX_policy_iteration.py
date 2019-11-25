"""
Solving environment using Policy Itertion.
Author : Wei Feng
"""
import numpy as np
import pandas as pd
import gym
from gym import wrappers
from stocks_env import StocksEnv
import time
from HX_maze import generate_random_map, FrozenLakeEnv

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

def run_episode_stock(env, policy, gamma = 1.0, render = False):
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
    obs = 11  # that's what usually happens. for PI/VI for the daily trading, obs has to be set to 0
    total_reward = 0
    step_idx = 0

    while True:
        if render:
            env.render()
        actual_obs_not_using, reward, done , _ = env.step(int(policy[obs]))
        obs += 1
        # print('obs:', obs, 'r:', reward, 'done:', done)
        # time.sleep(0.3)

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

def evaluate_policy_stock(env, policy, gamma = 1.0,  n = 1000):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode_stock(env, policy, gamma = gamma, render = False)
            for _ in range(n)]
    return np.mean(scores)


class PI:
    def __init__(self, env):
        self.env = env

    def policy_evaluation(self, policy, gamma):
        V = np.zeros(self.env.nS)
        THETA = 1e-10
        delta = float("inf")
        round_num = 0

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
        action_values = np.zeros(self.env.nA)
        for a in range(self.env.nA):
            for prob, next_state, reward, done in self.env.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state])
        return np.argmax(action_values), np.max(action_values)

    def optimize(self, gamma = 1):
        policy = np.tile(np.eye(self.env.nA)[1], (self.env.nS, 1))
        # print("self.env.nA", self.env.nA)
        is_stable = False

        round_num = 0

        while not is_stable:
            is_stable = True
            start_time = time.time()
            round_num += 1

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
            print('round_num/time:', round_num, time.time() - start_time)

        policy = [np.argmax(policy[s]) for s in range(self.env.nS)]
        print("PI policy:", policy)
        return policy


if __name__ == '__main__':
    env_name  = 'FrozenLake8x8-v0'
    env = gym.make(env_name)

    pi = PI(env)
    optimal_policy = pi.optimize(gamma=1)
    policy_score = evaluate_policy(env, optimal_policy, n=1000)
    print('Policy average score = ', policy_score)
    '''===========stocks==========='''
    env_AT = StocksEnv(df=pd.read_csv('SPY.csv'),frame_bound=(50, 100), window_size=10)
    print(env_AT.nA)
    print(env_AT.nS)
    pi_AT = PI(env_AT)
    optimal_policy_AT = pi_AT.optimize(gamma=1)
    policy_score = evaluate_policy_stock(env_AT, optimal_policy_AT, n=1000)
    print('Policy average score = ', policy_score)
