

import abc
import pandas as pd

from enum import Enum

from lib.data.providers.BaseDataProvider import BaseDataProvider

class ExchangeMode(Enum):
    TRAIN = 0
    TEST = 1
    PAPER = 2
    LIVE = 3

class BaseExchange(object, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        self.product_id: str = kwargs.get('product_id', 'BTC-USD')

        self.base_precision: int = kwargs.get('base_precision', 2)
        self.asset_precision: int = kwargs.get('asset_precision', 8)
        self.min_cost_limit: float = kwargs.get('min_cost_limit', 1E-3)
        self.min_amount_limit: float = kwargs.get('min_amount_limit', 1E-3)

    @abc.abstractmethod
    def get_account_history(self):
        raise NotImplementedError

    @abc.abstractmethod
    def price(self):
        raise NotImplementedError

    @abc.abstractmethod
    def buy(self, amount):
        raise NotImplementedError

    @abc.abstractmethod
    def sell(self, amount):
        raise NotImplementedError


class DummyExchange(BaseExchange):

    def __init__(self, data_provider: BaseDataProvider, initial_balance: int = 10000,
                commission: float = 0.0025, **kwargs):
                BaseExchange.__init__(self, **kwargs)
                self.initial_balance = round(initial_balance, self.base_precision)
                self.commission = commission
                self.data_provider = data_provider

                self.current_ohlcv = self.data_provider.next_ohlcv()
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

    def get_account_history(self):
        return self.account_history

    def price(self, ohlcv_key: str = 'Close'):
        return float(self.current_ohlcv[ohlcv_key])

    def buy(self, amount):
        asset_bought = 0
        asset_sold = 0
        cost_of_asset = 0
        revenue_from_sold = 0

        if self.balance >= self.min_cost_limit:
            buy_price = round(self.price() * (1 + self.commission), self.base_precision)
            cost_of_asset = round(self.balance * amount, self.base_precision)
            asset_bought = round(cost_of_asset / buy_price, self.asset_precision)

            self.last_bought = self.current_step
            self.asset_held += asset_bought
            self.balance -= cost_of_asset

            self.trades.append({'step': self.current_step, 'amount': asset_bought,
                                'total': cost_of_asset, 'type': 'buy'})
            current_net_worth = round(self.balance + self.asset_held * current_price, self.base_precision)
            self.net_worths.append(current_net_worth)

            self.account_history = self.account_history.append({
                'balance': self.balance,
                'asset_bought': asset_bought,
                'cost_of_asset': cost_of_asset,
                'asset_sold': asset_sold,
                'revenue_from_sold': revenue_from_sold,
            }, ignore_index=True)
        self.current_step += 1
        self.current_ohlcv = self.data_provider.next_ohlcv()

    def sell(self, amount):

        sell_price = round(self.price() * (1 - self.commission), self.base_precision)
        asset_sold = round(self.asset_held * amount, self.asset_precision)
        revenue_from_sold = round(asset_sold * sell_price, self.base_precision)

        self.last_sold = self.current_step
        self.asset_held -= asset_sold
        self.balance += revenue_from_sold

        self.trades.append({'step': self.current_step, 'amount': asset_sold,
                            'total': revenue_from_sold, 'type': 'sell'})
        current_net_worth = round(self.balance + self.asset_held * current_price, self.base_precision)
        self.net_worths.append(current_net_worth)

        self.account_history = self.account_history.append({
            'balance': self.balance,
            'asset_bought': asset_bought,
            'cost_of_asset': cost_of_asset,
            'asset_sold': asset_sold,
            'revenue_from_sold': revenue_from_sold,
        }, ignore_index=True)
        self.current_step += 1
        self.current_ohlcv = self.data_provider.next_ohlcv()

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
