"""
Solving FrozenLake8x8 environment using Value-Itertion.
Author : Moustafa Alzantot (malzantot@ucla.edu)
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

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                # print next_sr
                # print "p=", p
                # print "s_=", s_
                # print "r=", r
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma = 1.0):
    """ Value-iteration algorithm
    p, s_, r, _ are:
    probability, next_state, reward, and if this is the target or not.
    """
    v = np.zeros(env.nS)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        print ('iteration# %d.' % (i + 1))
        # print np.reshape(v, (8, 8))
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p*(r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
            aaa = [p*(r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][a]]
            # print "aaa:", aaa
            # print "state:", s, "q_sa:", q_sa
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break

    return v


if __name__ == '__main__':
    env_name  = 'FrozenLake8x8-v0'
    gamma = 0.99
    env = gym.make(env_name)
    optimal_v = value_iteration(env, gamma)
    policy = extract_policy(optimal_v, gamma)
    policy_score = evaluate_policy(env, policy, n=1000)
    print('Policy average score = ', policy_score)