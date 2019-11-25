
import numpy as np
import gym
from gym import wrappers
import value_iteration_wfeng
import policy_iteration_wfeng

if __name__ == '__main__':
    env_name  = 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    vi = value_iteration_wfeng.Agent(env)
    optimal_policy = vi.optimize()
    print optimal_policy




    # optimal_v = value_iteration(env, gamma = 1.0);
    # policy = extract_policy(optimal_v, gamma = 1.0)
    # policy_score = evaluate_policy(env, policy, gamma = 1.0, n=1000)
    # print('Policy average score = ', policy_score)