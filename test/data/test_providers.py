import pytest

from lib.data.providers.dates import ProviderDateFormat
from lib.data.providers import StaticDataProvider


@pytest.fixture
def csv_provider():
    data_columns = {'Date': 'Date', 'Open': 'Open', 'High': 'High',
                    'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume BTC'}
    provider = StaticDataProvider(
        date_format=ProviderDateFormat.DATETIME_HOUR_12, csv_data_path="data/input/coinbase_hourly.csv", data_columns=data_columns)

    assert csv_provider is not None

    return provider


class TestPrepareData():
    def test_column_map(self, csv_provider):
        ohlcv = csv_provider.historical_ohlcv()

        expected = ['Date', 'Open', 'High',
                    'Low', 'Close', 'Volume', 'Timestamp']

        assert (ohlcv.columns == expected).all()

    def test_date_sort(self, csv_provider):
        ohlcv = csv_provider.historical_ohlcv()

        timestamps = ohlcv['Date'].values
        sorted_timestamps = sorted(timestamps.copy())

        assert (timestamps == sorted_timestamps).all()
