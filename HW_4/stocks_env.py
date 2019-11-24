import numpy as np

from trading_env import TradingEnv, Actions, Positions

import time
class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size)

        # self.trade_fee_bid_percent = 0.01  # unit.   Original
        # self.trade_fee_ask_percent = 0.005  # unit. Original

        self.nA = 2
        self.nS = len(self.prices)
        self.P = {s : {a : [] for a in range(self.nA)} for s in range(self.nS)}
        '''[(0.3333333333333333, 0, 0.0, False)'''

        '''prob/next_state/reward/done'''
        '''[(1.0, array([], shape=(0, 2), dtype=float64), -3.1200000000000045, False)]'''
        # self._current_tick = 0
        # observation = self.signal_features[(self._current_tick-self.window_size):self._current_tick]

        # def _get_observation(self):
        #     return self.signal_features[(self._current_tick - self.window_size):self._current_tick]


        # print('self.signal_features:\n',self.signal_features)
        for s in range(self.nS):

                for a in range(self.nA):
                    self._current_tick = s
                    li = self.P[s][a]
                    next_tick = self._current_tick + 1

                    # next_state = self.signal_features[(next_tick - self.window_size):next_tick]
                    next_state = s + 1
                    if s >= self.window_size:
                        step_reward = self._calculate_reward(a)
                    else:
                        step_reward = 0

                    if next_tick == self.nS -1:
                        li.append((1.0, next_state, step_reward, True))
                    else:
                        if s == self.nS-1:
                            li.append((1.0, 59, 0, True))
                        else:
                            li.append((1.0, next_state, step_reward, False))
                    # print("====li========\n", li)

        self.trade_fee_bid_percent = 0.00  # HX
        self.trade_fee_ask_percent = 0.00  # HX


    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features

    #
    # def _calculate_reward(self, action):   # this is the original, which calculate reward based on each trade
    #     step_reward = 0
    #
    #     trade = False
    #     if ((action == Actions.Buy.value and self._position == Positions.Short) or
    #         (action == Actions.Sell.value and self._position == Positions.Long)):
    #         trade = True
    #
    #     if trade:
    #         current_price = self.prices[self._current_tick]
    #         last_trade_price = self.prices[self._last_trade_tick]
    #         price_diff = current_price - last_trade_price
    #
    #         if self._position == Positions.Long:
    #             step_reward += price_diff
    #
    #     return step_reward

    def _calculate_reward(self, action):   # HX version, calculate reward on a daily basis
        # using the observation of self.current_tick decide today's action and the very end of day, then use tomorrow's
        # (i.e. self.current_tick + 1) observation to decide how is the reward of today.

        step_reward = 0
        if self._current_tick == self._end_tick:     # then there will be tomorrow's price, nor reward
            step_reward = 0
        else:
            tomorrows_price = self.prices[self._current_tick+1]
            current_price = self.prices[self._current_tick]
            price_diff = tomorrows_price - current_price

            if action == 1:
                step_reward += price_diff
            elif action == 0:
                step_reward -= price_diff

        return step_reward

    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price


    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
