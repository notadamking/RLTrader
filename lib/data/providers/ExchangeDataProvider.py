import ccxt
import pandas as pd

from typing import Tuple
from datetime import datetime

from lib.data.providers.dates import ProviderDateFormat
from lib.data.providers import BaseDataProvider


class ExchangeDataProvider(BaseDataProvider):
    _current_index = '2018-01-01T00:00:00Z'
    _has_loaded_historical = False

    def __init__(self,
                 exchange_name: str = 'binance',
                 symbol_pair: str = 'BTC/USDT',
                 timeframe: str = '1h',
                 start_date: datetime = None,
                 date_format: ProviderDateFormat = ProviderDateFormat.TIMESTAMP_MS,
                 data_frame: pd.DataFrame = None,
                 **kwargs):
        BaseDataProvider.__init__(self, date_format, **kwargs)

        self.exchange_name = exchange_name
        self.symbol_pair = symbol_pair
        self.timeframe = timeframe
        self.data_frame = data_frame
        self.start_date = start_date

        self.kwargs = kwargs
        self._has_loaded_historical = kwargs.get('_has_loaded_historical', False)

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

    def split_data_train_test(self, train_split_percentage: float = 0.8) -> Tuple[BaseDataProvider, BaseDataProvider]:
        params = {
            'exchange_name': self.exchange_name,
            'symbol_pair': self.symbol_pair,
            'timeframe': self.timeframe,
            'start_date': self.start_date,
            'date_format': self.date_format,
            **self.kwargs
        }

        if self._has_loaded_historical:
            train_len = int(train_split_percentage * len(self.data_frame))

            train_df = self.data_frame[:train_len].copy()
            test_df = self.data_frame[train_len:].copy()

            train_provider = ExchangeDataProvider(data_frame=train_df, **params, _has_loaded_historical=True)
            test_provider = ExchangeDataProvider(data_frame=test_df, **params, _has_loaded_historical=True)

            return train_provider, test_provider
        else:
            train_provider = ExchangeDataProvider(data_frame=self.data_frame, **params)
            test_provider = ExchangeDataProvider(**params)

    def _load_historical_ohlcv(self) -> pd.DataFrame:
        self.data_frame = pd.DataFrame(None, columns=self.in_columns)
        since = self.exchange.parse8601(self.start_date)

        while since < self.exchange.milliseconds():
            data = self.exchange.fetchOHLCV(symbol=self.symbol_pair, timeframe=self.timeframe, since=since)

            if len(data):
                since = data[len(data) - 1]['timestamp']
                self.data_frame = self.data_frame.append(data, ignore_index=True)

        self.data_frame = self.prepare_data(self.data_frame)
        self._current_index = 0
        self._has_loaded_historical = True

        return self.data_frame

    def historical_ohlcv(self):
        if self._has_loaded_historical is False:
            self._load_historical_ohlcv()

        return self.data_frame

    def has_next_ohlcv(self) -> bool:
        return True

    def reset_ohlcv_index(self, index: datetime = '2018-01-01T00:00:00Z'):
        self._current_index = index

    def next_ohlcv(self) -> pd.DataFrame:
        if self._has_loaded_historical:
            frame = self.data_frame[self._current_index]

            self._current_index += 1

            return frame

        data = self.exchange.fetchOHLCV(symbol=self.symbol_pair, timeframe=self.timeframe,
                                        since=self._current_index, limit=1)

        if len(data):
            self._current_index = data[len(data) - 1]['timestamp']
            frame = pd.DataFrame(data, columns=self.in_columns)
            frame = self.prepare_data(frame)

            if self.data_frame is None:
                self.data_frame = pd.DataFrame(None, columns=self.columns)

            self.data_frame = self.data_frame.append(frame, ignore_index=True)

            return frame

        return None
