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
from lib.exchange.exchanges import BaseExchange, DummyExchange, ExchangeMode
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
                 exchange: BaseExchange = DummyExchange,
                 reward_strategy: BaseRewardStrategy = IncrementalProfit,
                 **kwargs):
        super(TradingEnv, self).__init__()

        self.logger = kwargs.get('logger', init_logger(__name__, show_debug=kwargs.get('show_debug', True)))

        self.data_provider = data_provider
        self.exchange = DummyExchange(data_provider) if exchange == DummyExchange else exchange

        self.reward_strategy = reward_strategy()

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

    def _take_action(self, action: int):
        action_type: TradingEnvAction = TradingEnvAction(action % 3)
        action_amount = float(1 / (action % 8 + 1))

        if action_type == TradingEnvAction.BUY:
            self.exchange.buy(action_amount)

        elif action_type == TradingEnvAction.SELL:
            self.exchange.sell(action_amount)

    def _done(self):
        lost_90_percent_net_worth = float(self.exchange.net_worths[-1]) < (self.exchange.initial_balance / 10)
        has_next_frame = self.data_provider.has_next_ohlcv()

        return lost_90_percent_net_worth or not has_next_frame

    def _reward(self):
        reward = self.reward_strategy.get_reward(observations=self.observations,
                                                 net_worths=self.exchange.net_worths,
                                                 account_history=self.exchange.account_history,
                                                 last_bought=self.exchange.last_bought,
                                                 last_sold=self.exchange.last_sold,
                                                 current_price=self.exchange.price())

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
            scaled_history = log_and_difference(self.exchange.get_account_history(), inplace=False)
        else:
            scaled_history = self.exchange.get_account_history()

        if self.normalize_obs:
            scaled_history = max_min_normalize(scaled_history, inplace=False)

        obs = np.insert(obs, len(obs), scaled_history.values[-1], axis=0)

        obs = np.reshape(obs.astype('float16'), self.obs_shape)
        obs[np.bitwise_not(np.isfinite(obs))] = 0

        return obs

    def reset(self):
        self.data_provider.reset_ohlcv_index()

        assert isinstance(self.exchange, DummyExchange)

        self.exchange.reset()
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
            account_history = self.exchange.account_history
            self.logger.info('Price: ' + str(self.exchange.price()))
            self.logger.info('Bought: ' + str(account_history[2][self.current_step]))
            self.logger.info('Sold: ' + str(account_history[4][self.current_step]))
            self.logger.info('Net worth: ' + str(self.exchange.net_worths[-1]))

        elif mode == 'human':
            if self.viewer is None:
                self.viewer = TradingChart(self.data_provider.data_frame)

            self.viewer.render(self.current_step,
                               self.exchange.net_worths,
                               self.render_benchmarks,
                               self.exchange.trades)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
