import gym
import pandas as pd
import numpy as np
from gym import spaces
from sklearn import preprocessing
from fbprophet import Prophet

from render.BitcoinTradingGraph import BitcoinTradingGraph
from util.indicators import add_indicators
from util.suppress_stan import suppress_stdout_stderr


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
        self.df, n_indicators = add_indicators(self.df.reset_index())

        self.n_forecast_periods = kwargs.get('n_forecast_periods', 10)
        self.obs_shape = (1, 11 + n_indicators +
                          (3 * self.n_forecast_periods))

        # Actions of the format Buy 1/4, Sell 3/4, Hold (amount ignored), etc.
        self.action_space = spaces.Discrete(12)

        # Observes the price action, indicators, account action, price forecasts
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.obs_shape, dtype=np.float16)

    def _next_observation(self):
        scaled_df = self.df[['Open', 'High', 'Low',
                             'Close', 'Volume BTC', 'Volume USD']].copy().values[:self.current_step + 2].astype('float64')
        scaled_df = self.scaler.fit_transform(scaled_df)
        scaled_df = pd.DataFrame(scaled_df, columns=['Open', 'High', 'Low',
                                                     'Close', 'Volume BTC', 'Volume USD'])

        obs = scaled_df.values[self.current_step].astype('float16')

        past_df = self.df[['Date', 'Close']][:self.current_step + 2].copy().rename(
            index=str, columns={'Date': 'ds', 'Close': 'y'})

        with suppress_stdout_stderr():
            prophet = Prophet()
            prophet.fit(past_df, iter=200)
            future = prophet.make_future_dataframe(
                periods=self.n_forecast_periods)
            forecast = prophet.predict(
                future)[['yhat', 'yhat_lower', 'yhat_upper']][-self.n_forecast_periods:]

        obs = np.insert(obs, len(obs), forecast.values.ravel(), axis=0)

        scaled_history = self.scaler.fit_transform(self.account_history)

        obs = np.insert(
            obs, len(obs), scaled_history[:, self.current_step], axis=0)

        obs = np.reshape(obs, self.obs_shape)

        return obs

    def _get_current_price(self):
        return self.df['Close'].values[self.current_step]

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

            print('Bought ' + str(btc_bought) + ' BTC for $' + str(cost))

            self.btc_held += btc_bought
            self.balance -= cost
        elif action_type == 1:
            price = current_price * (1 - self.commission)
            btc_sold = self.btc_held * amount
            sales = btc_sold * price

            print('Sold ' + str(btc_sold) + ' BTC for $' + str(sales))

            self.btc_held -= btc_sold
            self.balance += sales
        else:
            print('Hold')

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
        ], axis=1).astype('float64')

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
        done = self.net_worth <= 0 or self.current_step == len(self.df) - 1

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
