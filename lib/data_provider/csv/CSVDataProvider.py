import pandas as pd

from lib.data_provider.base.BaseDataProvider import BaseDataProvider, ProviderDateFormat


class CSVDataProvider(BaseDataProvider):
    __current_idx = 0

    def __init__(self, data_path: str, date_format: ProviderDateFormat, __data_columns: list = None, __date_column: str = None):
        BaseDataProvider.__init__(self, __data_columns, __date_column)

        self.data_path = data_path
        self.date_format = date_format

        self.data_frame = pd.read_csv(self.data_path)
        self.data_frame = self.data_frame[self.__data_columns]
        self.data_frame = self.prepare_data(self.data_frame)

    def __date_format(self):
        return self.date_format

    def historical_ohclv(self):
        return self.data_frame

    def next(self):
        frame = self.data_frame[self.__current_idx]

        self.__current_idx += 1

        return frame
