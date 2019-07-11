import gym
import pandas as pd
import numpy as np

from gym import spaces
from enum import Enum
from typing import List, Dict

from lib.env.render import TradingChart
from lib.env.reward import BaseRewardStrategy, IncrementalProfit, WeightedUnrealizedProfit
from lib.env.trade import BaseTradeStrategy, SimulatedTradeStrategy
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
                 trade_strategy: BaseTradeStrategy = SimulatedTradeStrategy,
                 initial_balance: int = 10000,
                 commissionPercent: float = 0.25,
                 maxSlippagePercent: float = 2.0,
                 **kwargs):
        super(TradingEnv, self).__init__()

        self.logger = kwargs.get('logger', init_logger(__name__, show_debug=kwargs.get('show_debug', True)))

        self.base_precision: int = kwargs.get('base_precision', 2)
        self.asset_precision: int = kwargs.get('asset_precision', 8)
        self.min_cost_limit: float = kwargs.get('min_cost_limit', 1E-3)
        self.min_amount_limit: float = kwargs.get('min_amount_limit', 1E-3)

        self.initial_balance = round(initial_balance, self.base_precision)
        self.commissionPercent = commissionPercent
        self.maxSlippagePercent = maxSlippagePercent

        self.data_provider = data_provider
        self.reward_strategy = reward_strategy()
        self.trade_strategy = trade_strategy(commissionPercent=self.commissionPercent,
                                             maxSlippagePercent=self.maxSlippagePercent,
                                             base_precision=self.base_precision,
                                             asset_precision=self.asset_precision,
                                             min_cost_limit=self.min_cost_limit,
                                             min_amount_limit=self.min_amount_limit)

        self.render_benchmarks: List[Dict] = kwargs.get('render_benchmarks', [])
        self.normalize_obs: bool = kwargs.get('normalize_obs', True)
        self.stationarize_obs: bool = kwargs.get('stationarize_obs', True)
        self.normalize_rewards: bool = kwargs.get('normalize_rewards', False)
        self.stationarize_rewards: bool = kwargs.get('stationarize_rewards', True)

        self.n_discrete_actions: int = kwargs.get('n_discrete_actions', 24)
        self.action_space = spaces.Discrete(self.n_discrete_actions)

        self.n_features = 6 + len(self.data_provider.columns)
        self.obs_shape = (1, self.n_features)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float16)

        self.observations = pd.DataFrame(None, columns=self.data_provider.columns)

    def _current_price(self, ohlcv_key: str = 'Close'):
        return float(self.current_ohlcv[ohlcv_key])

    def _get_trade(self, action: int):
        n_action_types = 3
        n_amount_bins = int(self.n_discrete_actions / n_action_types)

        action_type: TradingEnvAction = TradingEnvAction(action % n_action_types)
        action_amount = float(1 / (action % n_amount_bins + 1))

        amount_asset_to_buy = 0
        amount_asset_to_sell = 0

        if action_type == TradingEnvAction.BUY and self.balance >= self.min_cost_limit:
            price_adjustment = (1 + (self.commissionPercent / 100)) * (1 + (self.maxSlippagePercent / 100))
            buy_price = round(self._current_price() * price_adjustment, self.base_precision)
            amount_asset_to_buy = round(self.balance * action_amount / buy_price, self.asset_precision)
        elif action_type == TradingEnvAction.SELL and self.asset_held >= self.min_amount_limit:
            amount_asset_to_sell = round(self.asset_held * action_amount, self.asset_precision)

        return amount_asset_to_buy, amount_asset_to_sell

    def _take_action(self, action: int):
        amount_asset_to_buy, amount_asset_to_sell = self._get_trade(action)

        asset_bought, asset_sold, purchase_cost, sale_revenue = self.trade_strategy.trade(buy_amount=amount_asset_to_buy,
                                                                                          sell_amount=amount_asset_to_sell,
                                                                                          balance=self.balance,
                                                                                          asset_held=self.asset_held,
                                                                                          current_price=self._current_price)

        if asset_bought:
            self.asset_held += asset_bought
            self.balance -= purchase_cost

            self.trades.append({'step': self.current_step,
                                'amount': asset_bought,
                                'total': purchase_cost,
                                'type': 'buy'})
        elif asset_sold:
            self.asset_held -= asset_sold
            self.balance += sale_revenue

            self.reward_strategy.reset_reward()

            self.trades.append({'step': self.current_step,
                                'amount': asset_sold,
                                'total': sale_revenue,
                                'type': 'sell'})

        current_net_worth = round(self.balance + self.asset_held * self._current_price(), self.base_precision)
        self.net_worths.append(current_net_worth)
        self.account_history = self.account_history.append({
            'balance': self.balance,
            'asset_held': self.asset_held,
            'asset_bought': asset_bought,
            'purchase_cost': purchase_cost,
            'asset_sold': asset_sold,
            'sale_revenue': sale_revenue,
        }, ignore_index=True)

    def _done(self):
        lost_90_percent_net_worth = float(self.net_worths[-1]) < (self.initial_balance / 10)
        has_next_frame = self.data_provider.has_next_ohlcv()

        return lost_90_percent_net_worth or not has_next_frame

    def _reward(self):
        reward = self.reward_strategy.get_reward(current_step=self.current_step,
                                                 current_price=self._current_price,
                                                 observations=self.observations,
                                                 account_history=self.account_history,
                                                 net_worths=self.net_worths)

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
        self.timestamps.append(pd.to_datetime(self.current_ohlcv.Date.item(), unit='s'))
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
        self.timestamps = []
        self.asset_held = 0
        self.current_step = 0

        self.reward_strategy.reset_reward()

        self.account_history = pd.DataFrame([{
            'balance': self.balance,
            'asset_held': self.asset_held,
            'asset_bought': 0,
            'purchase_cost': 0,
            'asset_sold': 0,
            'sale_revenue': 0,
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

        return obs, reward, done, {'net_worths': self.net_worths, 'timestamps': self.timestamps}

    def render(self, mode='human'):
        if mode == 'system':
            self.logger.info('Price: ' + str(self._current_price()))
            self.logger.info('Bought: ' + str(self.account_history['asset_bought'][self.current_step]))
            self.logger.info('Sold: ' + str(self.account_history['asset_sold'][self.current_step]))
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
