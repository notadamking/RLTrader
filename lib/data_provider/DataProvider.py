import ccxt

class DataProvider():

    def __init__(self, exchange):
        self.exchange = exchange

    def get_data(self, symbol, timeframe='5m'):
        self.exchange.fetchOHLCV(timeframe=timeframe)