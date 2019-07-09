import numpy as np
import pandas as pd

from typing import Tuple, List, Dict
from datetime import datetime
from abc import ABCMeta, abstractmethod

from lib.data.providers.dates import ProviderDateFormat


class BaseDataProvider(object, metaclass=ABCMeta):
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    in_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    custom_datetime_format = None

    @abstractmethod
    def __init__(self, date_format: ProviderDateFormat, **kwargs):
        self.date_format = date_format

        self.custom_datetime_format: str = kwargs.get('custom_datetime_format', None)

        data_columns: Dict[str, str] = kwargs.get('data_columns', None)

        if data_columns is not None:
            self.data_columns = data_columns
            self.columns = list(data_columns.keys())
            self.in_columns = list(data_columns.values())
        else:
            self.data_columns = dict(zip(self.columns, self.in_columns))

    @abstractmethod
    def split_data_train_test(self, train_split_percentage: float = 0.8) -> Tuple:
        raise NotImplementedError

    @abstractmethod
    def historical_ohlcv(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def reset_ohlcv_index(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def has_next_ohlcv(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def next_ohlcv(self) -> pd.DataFrame:
        raise NotImplementedError

    def prepare_data(self, data_frame: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        column_map = dict(zip(self.in_columns, self.columns))

        formatted = data_frame[self.in_columns]
        formatted = formatted.rename(index=str, columns=column_map)

        formatted = self._format_date_column(formatted, inplace=inplace)
        formatted = self._sort_by_date(formatted, inplace=inplace)

        return formatted

    def _sort_by_date(self, data_frame: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        if inplace is True:
            formatted = data_frame
        else:
            formatted = data_frame.copy()

        formatted = formatted.sort_values(self.data_columns['Date'])

        return formatted

    def _format_date_column(self, data_frame: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        if inplace is True:
            formatted = data_frame
        else:
            formatted = data_frame.copy()

        date_col = self.data_columns['Date']
        date_frame = formatted.loc[:, date_col]

        if self.date_format is ProviderDateFormat.TIMESTAMP_UTC:
            formatted[date_col] = date_frame.apply(
                lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M'))
            formatted[date_col] = pd.to_datetime(date_frame, format='%Y-%m-%d %H:%M')
        elif self.date_format is ProviderDateFormat.TIMESTAMP_MS:
            formatted[date_col] = pd.to_datetime(date_frame, unit='ms')
        elif self.date_format is ProviderDateFormat.DATETIME_HOUR_12:
            formatted[date_col] = pd.to_datetime(date_frame, format='%Y-%m-%d %I-%p')
        elif self.date_format is ProviderDateFormat.DATETIME_HOUR_24:
            formatted[date_col] = pd.to_datetime(date_frame, format='%Y-%m-%d %H')
        elif self.date_format is ProviderDateFormat.DATETIME_MINUTE_12:
            formatted[date_col] = pd.to_datetime(date_frame, format='%Y-%m-%d %I:%M-%p')
        elif self.date_format is ProviderDateFormat.DATETIME_MINUTE_24:
            formatted[date_col] = pd.to_datetime(date_frame, format='%Y-%m-%d %H:%M')
        elif self.date_format is ProviderDateFormat.DATE:
            formatted[date_col] = pd.to_datetime(date_frame, format='%Y-%m-%d')
        elif self.date_format is ProviderDateFormat.CUSTOM_DATIME:
            formatted[date_col] = pd.to_datetime(
                date_frame, format=self.custom_datetime_format, infer_datetime_format=True)
        else:
            raise NotImplementedError

        formatted[date_col] = formatted[date_col].values.astype(np.int64) // 10 ** 9

        return formatted
