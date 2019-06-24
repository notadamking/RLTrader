import numpy as np
from lib.data_provider import DataProvider
from lib.RLTrader import RLTrader
from ccxt import binance

np.warnings.filterwarnings('ignore')

if __name__ == '__main__':
    trader = RLTrader()
    data_provider = DataProvider(binance)

    trader.optimize(n_trials=1)
    trader.train(n_epochs=1,
                 test_trained_model=True,
                 render_trained_model=True)
