import multiprocessing
import numpy as np

from multiprocessing import Pool

np.warnings.filterwarnings('ignore')


def optimize_code(params):
    from lib.RLTrader import RLTrader

    trader = RLTrader(**params)
    trader.optimize()


if __name__ == '__main__':
    n_processes = multiprocessing.cpu_count()
    params = {'n_envs': n_processes}

    opt_pool = Pool(processes=n_processes)
    opt_pool.imap(optimize_code, [params for _ in range(n_processes)])

    from lib.RLTrader import RLTrader

    trader = RLTrader(**params)
    trader.train(test_trained_model=True, render_test_env=True, render_report=True)
