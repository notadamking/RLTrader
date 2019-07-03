import gym
import pandas as pd
import numpy as np

from gym import spaces

from lib.env.render import TradingChart
from lib.data.providers import BaseDataProvider
from lib.data.features.transform import max_min_normalize, log_and_difference
from lib.util.logger import init_logger


class TradingEnv(gym.Env):
    '''A reinforcement trading environment made for use with gym-enabled algorithms'''
    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    def __init__(self, data_provider: BaseDataProvider, initial_balance=10000, commission=0.0025, **kwargs):
        super(TradingEnv, self).__init__()

        self.logger = kwargs.get('logger', init_logger(__name__, show_debug=kwargs.get('show_debug', True)))

        self.data_provider = data_provider
        self.initial_balance = initial_balance
        self.commission = commission

        self.reward_fn = kwargs.get('reward_fn', self._reward_incremental_profit)
        self.benchmarks = kwargs.get('benchmarks', [])
        self.enable_stationarization = kwargs.get('enable_stationarization', True)

        self.action_space = spaces.Discrete(3)

        n_features = 5 + len(self.data_provider.columns)

        self.obs_shape = (1, n_features)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float16)
        self.observations = pd.DataFrame(None, columns=self.data_provider.columns)

    def _next_observation(self):
        self.current_ohlcv = self.data_provider.next_ohlcv()
        self.observations = self.observations.append(self.current_ohlcv, ignore_index=True)

        if self.enable_stationarization:
            observations = log_and_difference(self.observations)
        else:
            observations = self.observations

        observations = max_min_normalize(observations)

        obs = observations.values[-1]

        scaled_history = max_min_normalize(self.account_history)

        obs = np.insert(obs, len(obs), scaled_history.values[-1], axis=0)

        obs = np.reshape(obs.astype('float16'), self.obs_shape)
        obs[np.bitwise_not(np.isfinite(obs))] = 0

        return obs

    def _current_price(self):
        return float(self.current_ohlcv['Close'])

    def _take_action(self, action):
        current_price = self._current_price()

        btc_bought = 0
        btc_sold = 0
        cost_of_btc = 0
        revenue_from_sold = 0

        if action == 0:
            price = current_price * (1 + self.commission)
            btc_bought = self.balance / price
            cost_of_btc = self.balance

            self.last_bought = self.current_step
            self.btc_held += btc_bought
            self.balance -= cost_of_btc
        elif action == 1:
            price = current_price * (1 - self.commission)
            btc_sold = self.btc_held
            revenue_from_sold = btc_sold * price

            self.last_sold = self.current_step
            self.btc_held -= btc_sold
            self.balance += revenue_from_sold

        if btc_sold > 0 or btc_bought > 0:
            self.trades.append({'step': self.current_step,
                                'amount': btc_sold if btc_sold > 0 else btc_bought, 'total': revenue_from_sold if btc_sold > 0 else cost_of_btc,
                                'type': 'sell' if btc_sold > 0 else 'buy'})

        self.net_worths.append(self.balance + self.btc_held * current_price)

        self.account_history = self.account_history.append({
            'balance': self.balance,
            'btc_bought': btc_bought,
            'cost_of_btc': cost_of_btc,
            'btc_sold': btc_sold,
            'revenue_from_sold': revenue_from_sold,
        }, ignore_index=True)

    def _reward_incremental_profit(self, observations, net_worths, account_history, last_bought, last_sold, current_price):
        prev_balance = account_history['balance'].values[-2]
        curr_balance = account_history['balance'].values[-1]
        reward = 0

        if curr_balance > prev_balance:
            reward = net_worths[-1] - net_worths[last_bought]
        elif curr_balance < prev_balance:
            reward = observations['Close'].values[last_sold] - current_price

        return reward

    def _reward(self):
        reward = self.reward_fn(observations=self.observations,
                                net_worths=self.net_worths,
                                account_history=self.account_history,
                                last_bought=self.last_bought,
                                last_sold=self.last_sold,
                                current_price=self._current_price())

        return reward if np.isfinite(reward) else 0

    def _done(self):
        lost_90_percent_net_worth = float(self.net_worths[-1]) < (self.initial_balance / 10)
        has_next_frame = self.data_provider.has_next_ohlcv()

        return lost_90_percent_net_worth or not has_next_frame

    def reset(self):
        self.data_provider.reset_ohlcv_index()

        self.balance = self.initial_balance
        self.net_worths = [self.initial_balance]
        self.btc_held = 0
        self.current_step = 0
        self.last_bought = 0
        self.last_sold = 0

        self.account_history = pd.DataFrame([{
            'balance': self.balance,
            'btc_bought': 0,
            'cost_of_btc': 0,
            'btc_sold': 0,
            'revenue_from_sold': 0,
        }])
        self.trades = []

        return self._next_observation()

    def step(self, action):
        self._take_action(action)

        self.current_step += 1

        obs = self._next_observation()
        reward = self._reward()
        done = self._done()

        return obs, reward, done, {}

    def render(self, mode='human'):
        if mode == 'system':
            self.logger.info('Price: ' + str(self._current_price()))
            self.logger.info('Bought: ' + str(self.account_history[2][self.current_step]))
            self.logger.info('Sold: ' + str(self.account_history[4][self.current_step]))
            self.logger.info('Net worth: ' + str(self.net_worths[-1]))

        elif mode == 'human':
            if self.viewer is None:
                self.viewer = TradingChart(self.data_provider.data_frame)

            self.viewer.render(self.current_step, self.net_worths, self.benchmarks, self.trades)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
