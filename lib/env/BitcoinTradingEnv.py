import gym
import pandas as pd
import numpy as np

from gym import spaces
from statsmodels.tsa.statespace.sarimax import SARIMAX
from empyrical import sortino_ratio, sharpe_ratio, omega_ratio

from lib.env.render.BitcoinTradingGraph import BitcoinTradingGraph
from lib.util.transform import log_and_difference, max_min_normalize
from lib.util.indicators import add_indicators


class BitcoinTradingEnv(gym.Env):
    '''A Bitcoin trading environment for OpenAI gym'''
    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    def __init__(self, df, initial_balance=10000, commission=0.0025, reward_strategy='sortino', **kwargs):
        super(BitcoinTradingEnv, self).__init__()

        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_strategy = reward_strategy

        self.df = df.fillna(method='bfill').reset_index()
        self.stationary_df = self.df.copy()
        self.stationary_df = self.stationary_df[self.stationary_df.columns.difference([
                                                                                      'index', 'Date'])]
        self.stationary_df = log_and_difference(self.stationary_df,
                                                ['Open', 'High', 'Low', 'Close', 'Volume BTC', 'Volume USD'])

        self.benchmarks = kwargs.get('benchmarks', [])

        self.forecast_steps = kwargs.get('forecast_steps', 2)
        self.forecast_alpha = kwargs.get('forecast_alpha', 0.05)

        self.action_space = spaces.Discrete(3)

        n_features = 5 + len(self.df.columns) - 2
        n_prediction_features = (self.forecast_steps * 3)
        self.obs_shape = (1, n_features + n_prediction_features)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.obs_shape, dtype=np.float16)

    def _next_observation(self):
        current_idx = self.current_step + self.forecast_steps + 1

        scaled = self.stationary_df[:current_idx].values

        scaled = pd.DataFrame(scaled, columns=self.stationary_df.columns)
        scaled = max_min_normalize(scaled)

        obs = scaled.values[-1]

        forecast_model = SARIMAX(self.stationary_df['Close'][:current_idx].values,
                                 enforce_stationarity=False,
                                 simple_differencing=True)

        model_fit = forecast_model.fit(method='bfgs', disp=False)

        forecast = model_fit.get_forecast(steps=self.forecast_steps,
                                          alpha=self.forecast_alpha)

        obs = np.insert(obs, len(obs), forecast.predicted_mean, axis=0)
        obs = np.insert(obs, len(obs), forecast.conf_int().flatten(), axis=0)

        scaled_history = max_min_normalize(self.account_history)

        obs = np.insert(obs, len(obs), scaled_history.values[-1], axis=0)

        obs = np.reshape(obs.astype('float16'), self.obs_shape)
        obs[np.bitwise_not(np.isfinite(obs))] = 0

        return obs

    def _current_price(self):
        return self.df['Close'].values[self.current_step + self.forecast_steps]

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

            self.btc_held += btc_bought
            self.balance -= cost_of_btc
        elif action == 1:
            price = current_price * (1 - self.commission)
            btc_sold = self.btc_held
            revenue_from_sold = btc_sold * price

            self.btc_held -= btc_sold
            self.balance += revenue_from_sold

        if btc_sold > 0 or btc_bought > 0:
            self.trades.append({'step': self.current_step,
                                'amount': btc_sold if btc_sold > 0 else btc_bought, 'total': revenue_from_sold if btc_sold > 0 else cost_of_btc,
                                'type': 'sell' if btc_sold > 0 else 'buy'})

        self.net_worths.append(
            self.balance + self.btc_held * current_price)

        self.account_history = self.account_history.append({
            'balance': self.balance,
            'btc_bought': btc_bought,
            'cost_of_btc': cost_of_btc,
            'btc_sold': btc_sold,
            'revenue_from_sold': revenue_from_sold,
        }, ignore_index=True)

    def _reward(self):
        length = min(self.current_step, self.forecast_steps)
        returns = np.diff(self.net_worths[-length:])

        if np.count_nonzero(returns) < 1:
            return 0

        if self.reward_strategy == 'sortino':
            reward = sortino_ratio(
                returns, annualization=365*24)
        elif self.reward_strategy == 'sharpe':
            reward = sharpe_ratio(
                returns, annualization=365*24)
        elif self.reward_strategy == 'omega':
            reward = omega_ratio(
                returns, annualization=365*24)
        else:
            reward = returns[-1]

        return reward if np.isfinite(reward) else 0

    def _done(self):
        return self.net_worths[-1] < self.initial_balance / 10 or self.current_step == len(self.df) - self.forecast_steps - 1

    def reset(self):
        self.balance = self.initial_balance
        self.net_worths = [self.initial_balance]
        self.btc_held = 0
        self.current_step = 0

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
            print('Price: ' + str(self._current_price()))
            print(
                'Bought: ' + str(self.account_history[2][self.current_step]))
            print(
                'Sold: ' + str(self.account_history[4][self.current_step]))
            print('Net worth: ' + str(self.net_worths[-1]))

        elif mode == 'human':
            if self.viewer is None:
                self.viewer = BitcoinTradingGraph(self.df)

            self.viewer.render(self.current_step,
                               self.net_worths, self.benchmarks, self.trades)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
