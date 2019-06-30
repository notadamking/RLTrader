import ccxt
import pandas as pd

from lib.data_provider.base.BaseDataProvider import BaseDataProvider, ProviderDateFormat


class ExchangeDataProvider(BaseDataProvider):
    __current_idx = 0

    def __init__(self, exchange_name: str, symbol_pair: str, timeframe: str, __data_columns: list = None, __date_column: str = None):
        BaseDataProvider.__init__(self, __data_columns, __date_column)

        self.exchange_name = exchange_name
        self.symbol_pair = symbol_pair
        self.timeframe = timeframe

        self.data_frame = None

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

        if self.symbol_pair not in self.exchange.symbols:
            raise ModuleNotFoundError(
                f'The requested symbol {self.symbol_pair} is not available from {self.exchange_name}')

    def __date_format(self):
        return ProviderDateFormat.TIMESTAMP_MS

    def historical_ohclv(self):
        # TODO: Pagination

        data = self.exchange.fetchOHLCV(
            symbol=self.symbol_pair, timeframe=self.timeframe)

        self.data_frame = pd.DataFrame(data, columns=self.__data_columns)
        self.data_frame = self.data_frame[self.__data_columns]
        self.data_frame = self.prepare_data(self.data_frame)

        return self.data_frame

    def next(self):
        if self.data_frame is not None:
            frame = self.data_frame[self.__current_idx]

            self.__current_idx += 1
        else:
            raise NotImplementedError()  # TODO

        return frame
