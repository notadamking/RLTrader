import time
import ccxt
import pandas as pd

from typing import Dict
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

        self.exchange.enableRateLimit = True

        if self.symbol_pair not in self.exchange.symbols:
            raise ModuleNotFoundError(
                f'The requested symbol {self.symbol_pair} is not available from {self.exchange_name}')
            
        markets = self.exchange.load_markets()
        self.market = markets[symbol_pair]

    def historical_ohlcv(self):
        if self._has_loaded_historical is False:
            self._load_historical_ohlcv()

        return self.data_frame

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

    # Precisions as defined in the unified CCXT API
    def get_market_fiat_precision(self) -> int:
        return self.market['precision']['price']
    
    def get_market_coin_precision(self) -> int:
        return self.market['precision']['amount']
    
    def get_market_min_price_limit(self) -> int:
        return self.market['limits']['price']['min']
    
    def get_market_max_price_limit(self) -> int:
        return self.market['limits']['price']['max']
        
    def get_market_min_amount_limit(self) -> float:
        return self.market['limits']['amount']['min']

    def get_market_max_amount_limit(self) -> float:
        return self.market['limits']['amount']['max']
      
    def get_market_min_cost_limit(self) -> float:
        return self.market['limits']['cost']['min']
    
    def get_market_max_cost_limit(self) -> float:
        return self.market['limits']['cost']['max']

    def get_market_taker_fee(self) -> float:
        return self.market['taker']
        
    def get_market_maker_fee(self) -> float:
        return self.market['maker']