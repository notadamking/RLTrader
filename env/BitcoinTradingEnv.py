import gym
import logging
import pandas as pd
import numpy as np
from numpy import inf
from gym import spaces
from sklearn import preprocessing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from render.BitcoinTradingGraph import BitcoinTradingGraph
from util.stationarization import log_and_difference
from util.indicators import add_indicators


class BitcoinTradingEnv(gym.Env):
    """A Bitcoin trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human', 'system', 'none']}
    scaler = preprocessing.MinMaxScaler()
    viewer = None

    def __init__(self, df, initial_balance=10000, commission=0.0003, **kwargs):
        super(BitcoinTradingEnv, self).__init__()

        self.initial_balance = initial_balance
        self.commission = commission

        self.df = df.fillna(method='bfill')
        self.df = add_indicators(self.df.reset_index())
        self.stationary_df = log_and_difference(
            self.df, ['Open', 'High', 'Low', 'Close', 'Volume BTC', 'Volume USD'])

        self.n_forecasts = kwargs.get('n_forecasts', 10)
        self.confidence_interval = kwargs.get('confidence_interval', 0.95)
        self.obs_shape = (1, 5 + len(self.df.columns) -
                          2 + (self.n_forecasts * 3))

        # Actions of the format Buy 1/4, Sell 3/4, Hold (amount ignored), etc.
        self.action_space = spaces.Discrete(12)

        # Observes the price action, indicators, account action, price forecasts
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.obs_shape, dtype=np.float16)

    def _next_observation(self):
        features = self.stationary_df[self.stationary_df.columns.difference([
            'index', 'Date'])]

        scaled = features[:self.current_step + self.n_forecasts].values
        scaled[abs(scaled) == inf] = 0
        scaled = self.scaler.fit_transform(scaled.astype('float64'))
        scaled = pd.DataFrame(scaled, columns=features.columns)

        obs = scaled.values[-1]

        past_df = self.stationary_df['Close'][:
                                              self.current_step + self.n_forecasts]
        forecast_model = SARIMAX(past_df.values)
        model_fit = forecast_model.fit(
            method='bfgs', disp=False)
        forecast = model_fit.get_forecast(
            steps=self.n_forecasts, alpha=(1 - self.confidence_interval))

        obs = np.insert(obs, len(obs), forecast.predicted_mean, axis=0)
        obs = np.insert(obs, len(obs), forecast.conf_int().flatten(), axis=0)

        scaled_history = self.scaler.fit_transform(
            self.account_history.astype('float64'))

        obs = np.insert(obs, len(obs), scaled_history[:, -1], axis=0)

        obs = np.reshape(obs.astype('float16'), self.obs_shape)

        return obs

    def _get_current_price(self):
        return self.df['Close'].values[self.current_step + self.n_forecasts]

    def _take_action(self, action, current_price):
        action_type = int(action / 4)
        amount = 1 / (action % 4 + 1)

        btc_bought = 0
        btc_sold = 0
        cost = 0
        sales = 0

        if action_type == 0:
            price = current_price * (1 + self.commission)
            btc_bought = min(self.balance * amount /
                             price, self.balance / price)
            cost = btc_bought * price

            self.btc_held += btc_bought
            self.balance -= cost
        elif action_type == 1:
            price = current_price * (1 - self.commission)
            btc_sold = self.btc_held * amount
            sales = btc_sold * price

            self.btc_held -= btc_sold
            self.balance += sales

        if btc_sold > 0 or btc_bought > 0:
            self.trades.append({'step': self.current_step,
                                'amount': btc_sold if btc_sold > 0 else btc_bought, 'total': sales if btc_sold > 0 else cost,
                                'type': "sell" if btc_sold > 0 else "buy"})

        self.net_worth = self.balance + self.btc_held * current_price

        self.account_history = np.append(self.account_history, [
            [self.balance],
            [btc_bought],
            [cost],
            [btc_sold],
            [sales]
        ], axis=1)

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.btc_held = 0
        self.current_step = 0

        self.account_history = np.array([
            [self.balance],
            [0],
            [0],
            [0],
            [0]
        ])
        self.trades = []

        return self._next_observation()

    def step(self, action):
        current_price = self._get_current_price() + 0.01

        prev_net_worth = self.net_worth

        self._take_action(action, current_price)

        self.current_step += 1

        obs = self._next_observation()
        reward = self.net_worth - prev_net_worth
        done = self.net_worth < self.initial_balance / \
            10 or self.current_step == len(self.df) - self.n_forecasts - 1

        return obs, reward, done, {}

    def render(self, mode='human', **kwargs):
        if mode == 'system':
            print('Price: ' + str(self._get_current_price()))
            print(
                'Bought: ' + str(self.account_history[2][self.current_step]))
            print(
                'Sold: ' + str(self.account_history[4][self.current_step]))
            print('Net worth: ' + str(self.net_worth))

        elif mode == 'human':
            if self.viewer is None:
                self.viewer = BitcoinTradingGraph(
                    self.df, kwargs.get('title', None))

            self.viewer.render(self.current_step, self.net_worth, self.trades)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
