import ccxt
import pandas as pd
from lib.util.indicators import add_indicators
from lib.data_feed.IDataProvider import IDataProvider

class CcxtDataProvider(IDataProvider):
    __columns = ['Date', 'Open', 'High', 'Low', 'Close', 'VolumeFrom']
    __post_filters = []

    def __init__(self, exchange, symbol_pair: str, timeframe: int, unit: str):
        self.exchange = exchange
        self.symbol_pair = symbol_pair
        self.timeframe = timeframe
        self.unit = unit

    def get_data(self):
        #if self.exchange.has['fetchOHLCV']:
        #    raise Exception('No OHCLV for this exchange')

        try:
            exchange = getattr (ccxt, self.exchange)()
        except AttributeError:
            raise ModuleNotFoundError('''
                Exchange "{}" not found. Please check if the exchange is supported.
                '''.format(self.exchange))

        if not exchange.has["fetchOHLCV"]:
            raise AttributeError('Exchange "{}" does not support fetchOHLCV'.format(self.exchange))

        exchange.load_markets()
        if self.symbol_pair not in exchange.symbols:
            raise IOError('''
                The requested symbol ({}) is not available from {}
                '''.format(self.symbol_pair, self.exchange))

        data = exchange.fetchOHLCV(symbol=self.symbol_pair, timeframe='{}{}'.format(self.timeframe, self.unit))

        feature_df = pd.DataFrame(data)
        feature_df.columns = self.__columns
        feature_df['Date'] = pd.to_datetime(feature_df['Date'], unit='ms')
        feature_df['Date'] = feature_df['Date'].astype(str)
        feature_df['VolumeTo'] = feature_df['VolumeFrom'] * (feature_df['Open'] + feature_df['Close']) / 2
        feature_df = feature_df.sort_values(['Date'])
        feature_df = add_indicators(feature_df.reset_index())

        for filter in self.__post_filters:
            feature_df = filter(feature_df)

        return feature_df

    def register_post_filter(self, callable):
        self.__post_filters.append(callable)