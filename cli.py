import numpy as np
import multiprocessing

from lib.RLTrader import RLTrader
from lib.cli.RLTraderCLI import RLTraderCLI
from lib.util.logger import init_logger
from lib.cli.functions import download_data_async

np.warnings.filterwarnings('ignore')
trader_cli = RLTraderCLI()
args = trader_cli.get_args()


def run_concurrent_optimize():
    trader = RLTrader(**vars(args))
    trader.optimize(args.trials)


def concurrent_optimize():
    processes = []
    for i in range(args.parallel_jobs):
        processes.append(multiprocessing.Process(target=run_concurrent_optimize, args=()))

    print(processes)

    for p in processes:
        p.daemon = True
        p.start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    logger = init_logger(__name__, show_debug=args.debug)
    trader = RLTrader(**vars(args), logger=logger)

    if args.command == 'optimize':
        concurrent_optimize()
    elif args.command == 'train':
        trader.train(n_epochs=args.epochs)
    elif args.command == 'test':
        trader.test(model_epoch=args.model_epoch, should_render=args.no_render)
    elif args.command == 'update-static-data':
        download_data_async()
