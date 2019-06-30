import time
import ccxt
import pandas as pd

from typing import Dict
from datetime import datetime

from lib.data.provider.dates import ProviderDateFormat
from lib.data.provider import BaseDataProvider


class ExchangeDataProvider(BaseDataProvider):
    __current_index = '2018-01-01T00:00:00Z'

    def __init__(self,
                 exchange_name: str = 'binance',
                 symbol_pair: str = 'BTC/USDT',
                 timeframe: str = '1h',
                 start_date: datetime = None,
                 date_format: ProviderDateFormat = ProviderDateFormat.TIMESTAMP_MS,
                 **kwargs):
        BaseDataProvider.__init__(self, date_format, **kwargs)

        self.exchange_name = exchange_name
        self.symbol_pair = symbol_pair
        self.timeframe = timeframe

        self.data_frame = None
        self.start_date = start_date

        get_exchange_fn = getattr(ccxt, self.exchange_name)

        try:
            self.exchange = get_exchange_fn()
        except AttributeError:
            raise ModuleNotFoundError(
                f'Exchange {self.exchange_name} not found. Please check if the exchange is supported.')

        if not self.exchange.has['fetchOHLCV']:
            raise AttributeError(
                f'Exchange {self.exchange_name} does not support fetchOHLCV')

        self.exchange.load_markets()
        self.exchange.enableRateLimit = True

        if self.symbol_pair not in self.exchange.symbols:
            raise ModuleNotFoundError(
                f'The requested symbol {self.symbol_pair} is not available from {self.exchange_name}')

    def historical_ohlcv(self):
        if self.data_frame is None:
            self._load_historical_ohlcv()

        return self.data_frame

    def _load_historical_ohlcv(self) -> pd.DataFrame:
        self.data_frame = pd.DataFrame(None, columns=self._pre_columns())
        since = self.exchange.parse8601(self.start_date)

        while since < self.exchange.milliseconds():
            data = self.exchange.fetchOHLCV(
                symbol=self.symbol_pair, timeframe=self.timeframe, since=since)

            if len(data):
                since = data[len(data) - 1]['timestamp']
                self.data_frame = self.data_frame.append(
                    data, ignore_index=True)

        self.data_frame = self.prepare_data(self.data_frame)
        self.__current_index = 0

        return self.data_frame

    def reset_ohlcv_index(self, index: datetime = '2018-01-01T00:00:00Z'):
        self.__current_index = index

    def next_ohlcv(self) -> pd.DataFrame:
        if self.data_frame is not None:
            frame = self.data_frame[self.__current_index]

            self.__current_index += 1

            return frame

        data = self.exchange.fetchOHLCV(
            symbol=self.symbol_pair, timeframe=self.timeframe, since=self.__current_index, limit=1)

        if len(data):
            self.__current_index = data[len(data) - 1]['timestamp']
            data_frame = pd.DataFrame(data, columns=self._pre_columns())

            return self.prepare_data(data_frame)

        return None
