import gym
import pandas as pd
import numpy as np

from gym import spaces
from enum import Enum
from typing import List, Dict

from lib.env.render import TradingChart
from lib.env.reward import BaseRewardStrategy, IncrementalProfit
from lib.data.providers import BaseDataProvider
from lib.data.features.transform import max_min_normalize, mean_normalize, log_and_difference, difference
from lib.util.logger import init_logger


class TradingEnvAction(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2


class TradingEnv(gym.Env):
    '''A reinforcement trading environment made for use with gym-enabled algorithms'''
    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    def __init__(self,
                 data_provider: BaseDataProvider,
                 reward_strategy: BaseRewardStrategy = IncrementalProfit,
                 initial_balance: int = 10000,
                 commission: float = 0.0025,
                 **kwargs):
        super(TradingEnv, self).__init__()

        self.logger = kwargs.get('logger', init_logger(__name__, show_debug=kwargs.get('show_debug', True)))

        self.base_precision: int = kwargs.get('base_precision', 2)
        self.asset_precision: int = kwargs.get('asset_precision', 8)
        self.min_cost_limit: float = kwargs.get('min_cost_limit', 1E-3)
        self.min_amount_limit: float = kwargs.get('min_amount_limit', 1E-3)

        self.data_provider = data_provider
        self.reward_strategy = reward_strategy()
        self.initial_balance = round(initial_balance, self.base_precision)
        self.commission = commission

        self.render_benchmarks: List[Dict] = kwargs.get('render_benchmarks', [])
        self.normalize_obs: bool = kwargs.get('normalize_obs', True)
        self.stationarize_obs: bool = kwargs.get('stationarize_obs', True)
        self.normalize_rewards: bool = kwargs.get('normalize_rewards', False)
        self.stationarize_rewards: bool = kwargs.get('stationarize_rewards', True)

        n_discrete_actions: int = kwargs.get('n_discrete_actions', 24)
        self.action_space = spaces.Discrete(n_discrete_actions)

        n_features = 5 + len(self.data_provider.columns)
        self.obs_shape = (1, n_features)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float16)

        self.observations = pd.DataFrame(None, columns=self.data_provider.columns)

    def _current_price(self, ohlcv_key: str = 'Close'):
        return float(self.current_ohlcv[ohlcv_key])

    def _make_trade(self, action: int, current_price: float):
        action_type: TradingEnvAction = TradingEnvAction(action % 3)
        action_amount = float(1 / (action % 8 + 1))

        asset_bought = 0
        asset_sold = 0
        cost_of_asset = 0
        revenue_from_sold = 0

        if action_type == TradingEnvAction.BUY and self.balance >= self.min_cost_limit:
            buy_price = round(current_price * (1 + self.commission), self.base_precision)
            cost_of_asset = round(self.balance * action_amount, self.base_precision)
            asset_bought = round(cost_of_asset / buy_price, self.asset_precision)

            self.last_bought = self.current_step
            self.asset_held += asset_bought
            self.balance -= cost_of_asset

            self.trades.append({'step': self.current_step, 'amount': asset_bought,
                                'total': cost_of_asset, 'type': 'buy'})

        elif action_type == TradingEnvAction.SELL and self.asset_held >= self.min_amount_limit:
            sell_price = round(current_price * (1 - self.commission), self.base_precision)
            asset_sold = round(self.asset_held * action_amount, self.asset_precision)
            revenue_from_sold = round(asset_sold * sell_price, self.base_precision)

            self.last_sold = self.current_step
            self.asset_held -= asset_sold
            self.balance += revenue_from_sold

            self.trades.append({'step': self.current_step, 'amount': asset_sold,
                                'total': revenue_from_sold, 'type': 'sell'})

        return asset_bought, asset_sold, cost_of_asset, revenue_from_sold

    def _take_action(self, action: int):
        current_price = self._current_price()

        asset_bought, asset_sold, cost_of_asset, revenue_from_sold = self._make_trade(action, current_price)

        current_net_worth = round(self.balance + self.asset_held * current_price, self.base_precision)
        self.net_worths.append(current_net_worth)

        self.account_history = self.account_history.append({
            'balance': self.balance,
            'asset_bought': asset_bought,
            'cost_of_asset': cost_of_asset,
            'asset_sold': asset_sold,
            'revenue_from_sold': revenue_from_sold,
        }, ignore_index=True)

    def _done(self):
        lost_90_percent_net_worth = float(self.net_worths[-1]) < (self.initial_balance / 10)
        has_next_frame = self.data_provider.has_next_ohlcv()

        return lost_90_percent_net_worth or not has_next_frame

    def _reward(self):
        reward = self.reward_strategy.get_reward(observations=self.observations,
                                                 net_worths=self.net_worths,
                                                 account_history=self.account_history,
                                                 last_bought=self.last_bought,
                                                 last_sold=self.last_sold,
                                                 current_price=self._current_price())

        reward = float(reward) if np.isfinite(float(reward)) else 0

        self.rewards.append(reward)

        if self.stationarize_rewards:
            rewards = difference(self.rewards, inplace=False)
        else:
            rewards = self.rewards

        if self.normalize_rewards:
            mean_normalize(rewards, inplace=True)

        rewards = np.array(rewards).flatten()

        return float(rewards[-1])

    def _next_observation(self):
        self.current_ohlcv = self.data_provider.next_ohlcv()
        self.observations = self.observations.append(self.current_ohlcv, ignore_index=True)

        if self.stationarize_obs:
            observations = log_and_difference(self.observations, inplace=False)
        else:
            observations = self.observations

        if self.normalize_obs:
            observations = max_min_normalize(observations)

        obs = observations.values[-1]

        if self.stationarize_obs:
            scaled_history = log_and_difference(self.account_history, inplace=False)
        else:
            scaled_history = self.account_history

        if self.normalize_obs:
            scaled_history = max_min_normalize(scaled_history, inplace=False)

        obs = np.insert(obs, len(obs), scaled_history.values[-1], axis=0)

        obs = np.reshape(obs.astype('float16'), self.obs_shape)
        obs[np.bitwise_not(np.isfinite(obs))] = 0

        return obs

    def reset(self):
        self.data_provider.reset_ohlcv_index()

        self.balance = self.initial_balance
        self.net_worths = [self.initial_balance]
        self.asset_held = 0
        self.current_step = 0
        self.last_bought = 0
        self.last_sold = 0

        self.account_history = pd.DataFrame([{
            'balance': self.balance,
            'asset_bought': 0,
            'cost_of_asset': 0,
            'asset_sold': 0,
            'revenue_from_sold': 0,
        }])
        self.trades = []
        self.rewards = [0]

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

            self.viewer.render(self.current_step,
                               self.net_worths,
                               self.render_benchmarks,
                               self.trades)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
