import gym
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt
import pandas as pd
from stocks_env import StocksEnv



# env = gym.make('forex-v0', frame_bound=(50, 100), window_size=10)
# env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=10)
env = StocksEnv(df=pd.read_csv('STOCKS_GOOGL.csv'),frame_bound=(50, 100), window_size=10)
print("==========env information:========")
print("> shape:", env.shape)
print("> df.shape:", env.df.shape)
print("> prices.shape:", env.prices.shape)
print("> signal_features.shape:", env.signal_features.shape)
print("> max_possible_profit:", env.max_possible_profit())
print(env.action_space.sample())

observation = env.reset()
while True:
    action = env.action_space.sample()   # Short=0 and Long=1

    # action =-1  # Short=0 and Long=1

    # print('action:', action)
    observation, reward, done, info = env.step(action)
    print('steps:',action, reward)
    # env.render()
    if done:
        print("info:", info)
        break


plt.cla()
env.render_all()
plt.show()