import pandas as pd

from lib.data.provider.base.BaseDataProvider import BaseDataProvider, ProviderDateFormat


class StaticDataProvider(BaseDataProvider):
    __current_index = 0

    def __init__(self, date_format: ProviderDateFormat, data_frame: pd.DataFrame = None, csv_data_path: str = None, __date_column: str = None, __ohlcv_columns: list = None):
        BaseDataProvider.__init__(self, __date_column, __ohlcv_columns)

        self.date_format = date_format

        if data_frame is not None:
            self.data_frame = data_frame
        elif csv_data_path is not None:
            self.data_frame = pd.read_csv(csv_data_path)
        else:
            raise ValueError(
                'StaticDataProvider requires either a data_frame or csv_data_path argument.')

        self.data_frame = self.prepare_data(self.data_frame)

    def __date_format(self) -> ProviderDateFormat:
        return self.date_format

    def historical_ohlcv(self) -> pd.DataFrame:
        return self.data_frame

    def reset_ohlcv_index(self) -> int:
        self.__current_index = 0

    def next_ohlcv(self) -> pd.DataFrame:
        frame = self.data_frame[self.__current_index]

        self.__current_index += 1

        return frame
