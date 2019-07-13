import pandas as pd
import os

from stochastic.continuous import FractionalBrownianMotion
from typing import Tuple

from lib.data.providers.dates import ProviderDateFormat
from lib.data.providers import BaseDataProvider


class SimulatedDataProvider(BaseDataProvider):
    _current_index = 0

    def __init__(self, date_format: ProviderDateFormat, length: int = 2000, skip_prepare_data: bool = False, **kwargs):
        BaseDataProvider.__init__(self, date_format, **kwargs)

        self.length = length
        self.kwargs = kwargs

        hurst = kwargs.get('hurst', 0.68)
        base_price = kwargs.get('base_price', 3000)
        base_volume = kwargs.get('base_volume', 1000)

        fbm = FractionalBrownianMotion(t=1, hurst=hurst)

        dates = fbm.times(self.length)
        prices = fbm.sample(self.length)
        volumes = fbm.sample(self.length)

        price_frame = pd.DataFrame([], columns=['Date', 'Price'], dtype=float)
        volume_frame = pd.DataFrame([], columns=['Date', 'Volume'], dtype=float)

        price_frame['Date'] = pd.to_datetime(dates, unit="m")
        price_frame['Price'] = prices

        volume_frame['Date'] = pd.to_datetime(dates, unit="m")
        volume_frame['Volume'] = volumes

        price_frame.set_index('Date')
        price_frame.index = pd.to_datetime(price_frame.index, unit='s')

        volume_frame.set_index('Date')
        volume_frame.index = pd.to_datetime(price_frame.index, unit='s')

        ohlc = price_frame['Price'].resample('1min').ohlc()
        volume = volume_frame['Volume'].resample('1min').sum()

        self.data_frame = pd.DataFrame([], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        self.data_frame.reset_index(drop=True)

        ohlc.reset_index(drop=True)
        volume.reset_index(drop=True)

        self.data_frame['Date'] = ohlc.index
        self.data_frame['Open'] = ohlc['open'].values * base_price
        self.data_frame['High'] = ohlc['high'].values * base_price
        self.data_frame['Low'] = ohlc['low'].values * base_price
        self.data_frame['Close'] = ohlc['close'].values * base_price
        self.data_frame['Volume'] = volume.values * base_volume

        if not skip_prepare_data:
            self.data_frame = self.prepare_data(self.data_frame)

    @staticmethod
    def from_prepared(date_format: ProviderDateFormat, length: int = 100000, **kwargs):
        return SimulatedDataProvider(date_format=date_format, length=length, skip_prepare_data=True, **kwargs)

    def split_data_train_test(self, train_split_percentage: float = 0.8) -> Tuple[BaseDataProvider, BaseDataProvider]:
        train_len = int(train_split_percentage * len(self.data_frame))

        train_df = self.data_frame[:train_len].copy()
        test_df = self.data_frame[train_len:].copy()

        train_provider = SimulatedDataProvider.from_prepared(
            date_format=self.date_format, length=self.length ** self.kwargs)
        test_provider = SimulatedDataProvider.from_prepared(
            date_format=self.date_format, length=self.length ** self.kwargs)

        return train_provider, test_provider

    def historical_ohlcv(self) -> pd.DataFrame:
        return self.data_frame

    def has_next_ohlcv(self) -> bool:
        return self._current_index < len(self.data_frame)

    def reset_ohlcv_index(self) -> int:
        self._current_index = 0

    def next_ohlcv(self) -> pd.DataFrame:
        frame = self.data_frame[self.columns].values[self._current_index]
        frame = pd.DataFrame([frame], columns=self.columns)

        self._current_index += 1

        return frame
