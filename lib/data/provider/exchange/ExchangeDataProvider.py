import time
import ccxt
import pandas as pd

from datetime import datetime

from lib.data.provider.base.BaseDataProvider import BaseDataProvider, ProviderDateFormat


class ExchangeDataProvider(BaseDataProvider):
    __current_index = '2018-01-01T00:00:00Z'

    def __init__(self,
                 exchange_name: str = 'binance',
                 symbol_pair: str = 'BTC/USDT',
                 timeframe: str = '1h',
                 start_date: datetime = None,
                 __date_column: str = None,
                 __ohlcv_columns: list = None):
        BaseDataProvider.__init__(self, __date_column, __ohlcv_columns)

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

    def __date_format(self) -> ProviderDateFormat:
        return ProviderDateFormat.TIMESTAMP_MS

    def historical_ohlcv(self):
        if self.data_frame is None:
            self.__load_historical_ohlcv()

        return self.data_frame

    def __load_historical_ohlcv(self) -> pd.DataFrame:
        columns = self.__columns()

        self.data_frame = pd.DataFrame(None, columns=columns)
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
            data_frame = pd.DataFrame(data, columns=self.__columns())

            return self.prepare_data(data_frame)

        return None
