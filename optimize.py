import numpy as np

import multiprocessing
from lib.RLTrader import RLTrader

np.warnings.filterwarnings('ignore')


def optimize_code(params):
    trader = RLTrader(**params)
    trader.optimize()


if __name__ == '__main__':
    n_process = multiprocessing.cpu_count()
    params = {}

    processes = []
    for i in range(n_process):
        processes.append(multiprocessing.Process(target=optimize_code, args=(params,)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    trader = RLTrader(**params)
    trader.train(test_trained_model=True, render_trained_model=True)
