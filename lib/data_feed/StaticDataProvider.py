import os
import re
import inspect
import pandas as pd
from lib.util.indicators import add_indicators
from lib.data_feed.IDataProvider import IDataProvider

class StaticDataProvider(IDataProvider):
    __data_path = "data/input"

    def __init__(self, exchange, symbol_pair: str, timeframe: int, unit: str):
        self.exchange = exchange
        self.symbol_pair = symbol_pair
        self.number = timeframe
        self.unit = unit

    def get_data(self):
        dir = inspect.getfile(self.__class__)

        clean_symbol_pair = re.sub('[^0-9a-z]', '-', self.symbol_pair.lower(), flags=re.I)
        fn = "{}-{}{}-{}.csv".format(self.exchange, str(self.number), self.unit, clean_symbol_pair)
        data_path = os.path.join(os.path.realpath(os.path.join(dir, '../../../')), self.__data_path, fn)
        feature_df = pd.read_csv(data_path)
        feature_df = feature_df.drop(['Symbol'], axis=1)
        feature_df['Date'] = pd.to_datetime(feature_df['Date'], format='%Y-%m-%d %H:%M')
        feature_df['Date'] = feature_df['Date'].astype(str)
        feature_df = feature_df.sort_values(['Date'])
        feature_df = add_indicators(feature_df.reset_index())
        feature_df.to_csv(data_path + '2', sep=',', index=False)

        return feature_df
