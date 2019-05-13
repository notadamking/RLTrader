from alpha_vantage.cryptocurrencies import CryptoCurrencies
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
import pandas as pd

from src.utils.enum import TimeFrame
from src.utils.enum import TECHIND
from src.utils.enum import INTERVAL


class DataLoader:
    """
    API limit: 5 API requests per minute and 500 requests per day
    Get your free API key from
    https://www.alphavantage.co/support/#support

    FOREX, while supported, is not implemented.
    
    API details here:
    https://github.com/RomelTorres/alpha_vantage
    """
    def __init__(self, api_key):
        self.API_KEY = api_key
        self.cc = CryptoCurrencies(key=self.API_KEY, output_format='pandas')
        self.ti = TechIndicators(key=self.API_KEY, output_format='pandas')
        self.ts = TimeSeries(key=self.API_KEY, output_format='pandas')


    def get_local_data(self, path):
        df = pd.read_csv(path)
        return df.sort_values('Timestamp')

    def get_crypto(self, crypto_symbol):

        data, _ = self.cc.get_digital_currency_daily(symbol=crypto_symbol, market='CNY')

        return rename_data(data, crypto=True)

    def get_stock(self, stock:str, period = TimeFrame.TimeFrame.DAILY, full: bool = False):
        """
        Returns stock data and meta data of the ticker for the specified time frame
        :param stock: [ENUM]: Stock ticker
        :param period: [ENUM] DAILY, WEEKLY, MONTHLY
        :param full: Returns only last 100 ticks if False, otherwise full tick data set if True. False by default.
        :return: stock data, meta_data
        """
        if period is TimeFrame.TimeFrame.DAILY:
            if full:
                data, _  = self.ts.get_daily(symbol=stock.upper(), outputsize='full')
                return rename_data(data)
            else:
                data, _ = self.ts.get_daily(symbol=stock.upper(), outputsize='compact')
                return rename_data(data)

        if period is TimeFrame.TimeFrame.WEEKLY:
            if full:
                data, _ = data = self.ts.get_weekly(symbol=stock.upper(), outputsize='full')
                return rename_data(data)
            else:
                data, _  = self.ts.get_weekly(symbol=stock.upper(), outputsize='compact')
                return rename_data(data)

        if period is TimeFrame.TimeFrame.MONTHLY:
            if full:
                data, _  = self.ts.get_monthly(symbol=stock.upper(), outputsize='full')
                return rename_data(data)
            else:
                data, _ = self.ts.get_monthly(symbol=stock.upper(), outputsize='compact')
                return rename_data(data)

    def get_intraday_stock(self, stock: str, interval: INTERVAL = INTERVAL.INTERVAL.FIVE_MIN, full: bool = False):
        """
        Returns intraday tick data for the given stock in the given time interval
        :param stock: [ENUM] ticker
        :param interval: [ENUM] 1min, 5min, 15min, 30min, 60min
        :param full: Returns only last 100 ticks if False, otherwise full tick data set if True. False by default.
        :return: stock data, meta_data
        """

        if full:
            data, _  = self.ts.get_intraday(symbol=stock.upper(), interval=interval.value, outputsize='full')
            return rename_data(data)
        else:
            data, _  = self.ts.get_intraday(symbol=stock.upper(), interval=interval.value, outputsize='compact')
            return rename_data(data)

    def get_tech_indicator(self, stock="MSFT",
                           indicator=TECHIND.TECHIND.BBANDS,
                           interval = TimeFrame.TimeFrame.DAILY,
                           time_period: int = 20,
                           ):
        """
        Returns the technical indicator for the given stock ticker on the given interval
        :param indicator [ENUM]: @See TECHIND
        :param stock [ENUM]: Ticker
        :param interval: Daily, Weekly, Monthly. Set to Daily by default
        :param time_period: Nr of time units between two calculating points. Set to 20 by default.
        :return: pandas dataframe containing the technical indicator for all recorded trading days of the stock.
        """
        if indicator is TECHIND.TECHIND.BBANDS:
            data, _ = self.ti.get_bbands(symbol=stock.upper(), interval=interval.name.lower(), time_period=time_period)

        if indicator is TECHIND.TECHIND.SMA:
            data, _ = self.ti.get_sma(symbol=stock.upper(), interval=interval.name.lower(), time_period=time_period)

        if indicator is TECHIND.TECHIND.EMA:
            data, _ = self.ti.get_ema(symbol=stock.upper(), interval=interval.name.lower(), time_period=time_period)

        if indicator is TECHIND.TECHIND.WMA:
            data, _ = self.ti.get_wma(symbol=stock.upper(), interval=interval.name.lower(), time_period=time_period)

        if indicator is TECHIND.TECHIND.MACD:
            data, _ = self.ti.get_macd(symbol=stock.upper(), interval=interval.name.lower())

        if indicator is TECHIND.TECHIND.STOCH:
            data, _ = self.ti.get_stoch(symbol=stock.upper(), interval=interval.name.lower())

        if indicator is TECHIND.TECHIND.RSI:
            data, _ = self.ti.get_rsi(symbol=stock.upper(), interval=interval.name.lower(), time_period=time_period)

        if indicator is TECHIND.TECHIND.ADX:
            data, _ = self.ti.get_adx(symbol=stock.upper(), interval=interval.name.lower(), time_period=time_period)

        if indicator is TECHIND.TECHIND.CCI:
            data, _ = self.ti.get_cci(symbol=stock.upper(), interval=interval.name.lower(), time_period=time_period)

        if indicator is TECHIND.TECHIND.AROON:
            data, _ = self.ti.get_aroon(symbol=stock.upper(), interval=interval.name.lower(), time_period=time_period)

        if indicator is TECHIND.TECHIND.MOM:
            data, _ = self.ti.get_mom(symbol=stock.upper(), interval=interval.name.lower())

        if indicator is TECHIND.TECHIND.OBV:
            data, _ = self.ti.get_obv(symbol=stock.upper(), interval=interval.name.lower())

        return data


def rename_data(data, crypto=False):
    """
    :param data:
    :return:
    """
    data = __rename_column(data, "date", 'Date')
    data = __rename_column(data, "1. open", 'Open')
    data = __rename_column(data, "2. high", 'High')
    data = __rename_column(data, "3. low", 'Low')
    data = __rename_column(data, "4. close", 'Close')
    data = __rename_column(data, "5. volume", 'Volume')
    if crypto:
        data = __rename_column(data, "6. market cap (USD)", 'MarketCap')

    return data


def __rename_column(df, old_name, new_name):
    """
    Renames a column  in a pandas dataframe
    :param df: pandas dataframe
    :param old_name:
    :param new_name:
    :return: data frame with renamed column
    """
    return df.rename(index=str, columns={old_name: new_name})
