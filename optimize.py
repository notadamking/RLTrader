import numpy as np

from lib.RLTrader import RLTrader

np.warnings.filterwarnings('ignore')

if __name__ == '__main__':
    trader = RLTrader()

    trader.optimize()
    trader.train(test_trained_model=True, render_trained_model=True)
