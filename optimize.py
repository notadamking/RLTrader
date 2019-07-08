import os
import numpy as np

from multiprocessing.pool import ThreadPool

from lib.RLTrader import RLTrader

np.warnings.filterwarnings('ignore')


def optimize_code(params):
    trader = RLTrader(**params)
    trader.optimize()


if __name__ == '__main__':
    n_processes = os.cpu_count()
    params = {'n_envs': n_processes}

    opt_pool = ThreadPool(processes=n_processes)
    opt_pool.map(optimize_code, [params for _ in range(n_processes)])

    trader = RLTrader(**params)
    trader.train(test_trained_model=True, render_trained_model=True)
