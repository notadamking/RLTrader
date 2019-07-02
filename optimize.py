import numpy as np
import multiprocessing
from lib.RLTrader import RLTrader

np.warnings.filterwarnings('ignore')


def optimize_code(params):
    trader = RLTrader(**params)

    trader.optimize()


if __name__ == '__main__':
    # The no of process you wanna launch
    n_process = multiprocessing.cpu_count()
    # Database path param, recommended to replace with your own MySQL or PostgreSQL path
    params = {
        'params_db_path': 'sqlite:///data/params.db'
    }

    process = []
    # multiprocessing
    # creating processes
    for i in range(n_process):
        process.append(multiprocessing.Process(target=optimize_code, args=(params,)))

    # start processes
    for p in process:
        p.start()

    # join
    for p in process:
        p.join()


    trader = RLTrader(**params)

    trader.train(test_trained_model=True, render_trained_model=True)
