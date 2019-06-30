import numpy as np
import pandas as pd

from typing import List, Dict
from datetime import datetime
from abc import ABCMeta, abstractmethod

from lib.data.provider.dates import ProviderDateFormat


class BaseDataProvider(object, metaclass=ABCMeta):
    __data_columns = {'Date':  'Date',  'Open': 'Open', 'High': 'High',
                      'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}
    __column_map = {'Date':  'Date',  'Open': 'Open', 'High': 'High',
                    'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}
    __custom_datetime_format = None

    @abstractmethod
    def __init__(self, date_format: ProviderDateFormat, **kwargs):
        self.date_format = date_format

        self.__custom_datetime_format: str = kwargs.get(
            'custom_datetime_format', None)

        data_columns: Dict[str, str] = kwargs.get('data_columns', None)

        if data_columns is not None:
            self.__data_columns = data_columns
            self.__column_map = dict(
                zip(data_columns.values(), data_columns.keys()))

    @abstractmethod
    def historical_ohlcv(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def reset_ohlcv_index(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def next_ohlcv(self) -> pd.DataFrame:
        raise NotImplementedError

    def _columns(self) -> List[str]:
        return self.__data_columns.keys()

    def _pre_columns(self) -> List[str]:
        return self.__data_columns.values()

    def prepare_data(self, data_frame: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        formatted = data_frame[self._pre_columns()]
        formatted = formatted.rename(index=str, columns=self.__column_map)

        formatted = self._format_date_column(formatted, inplace=inplace)
        formatted = self._sort_by_date(formatted, inplace=inplace)

        return formatted

    def _sort_by_date(self, data_frame: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        if inplace is True:
            formatted = data_frame
        else:
            formatted = data_frame.copy()

        formatted = formatted.sort_values(['Timestamp'])

        return formatted

    def _format_date_column(self, data_frame: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        if inplace is True:
            formatted = data_frame
        else:
            formatted = data_frame.copy()

        date_col = formatted.loc[:, self.__data_columns['Date']]

        if self.date_format is ProviderDateFormat.TIMESTAMP_UTC:
            date_col = date_col.apply(
                lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M'))
            date_col = pd.to_datetime(date_col, format='%Y-%m-%d %H:%M')
        elif self.date_format is ProviderDateFormat.TIMESTAMP_MS:
            date_col = date_col = pd.to_datetime(date_col, unit='ms')
        elif self.date_format is ProviderDateFormat.DATETIME_HOUR_12:
            date_col = pd.to_datetime(date_col, format='%Y-%m-%d %I-%p')
        elif self.date_format is ProviderDateFormat.DATETIME_HOUR_24:
            date_col = pd.to_datetime(date_col, format='%Y-%m-%d %H')
        elif self.date_format is ProviderDateFormat.DATETIME_MINUTE_12:
            date_col = pd.to_datetime(date_col, format='%Y-%m-%d %I:%M-%p')
        elif self.date_format is ProviderDateFormat.DATETIME_MINUTE_24:
            date_col = pd.to_datetime(date_col, format='%Y-%m-%d %H:%M')
        elif self.date_format is ProviderDateFormat.DATE:
            date_col = pd.to_datetime(date_col, format='%Y-%m-%d')
        elif self.date_format is ProviderDateFormat.CUSTOM_DATIME:
            date_col = pd.to_datetime(
                date_col, format=self.__custom_datetime_format, infer_datetime_format=True)
        else:
            raise NotImplementedError

        data_frame['Timestamp'] = date_col.values.astype(np.int64) // 10 ** 9

        return formatted
