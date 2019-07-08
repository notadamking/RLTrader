import numpy as np

from multiprocessing.pool import ThreadPool

from lib.RLTrader import RLTrader
from lib.cli.RLTraderCLI import RLTraderCLI
from lib.util.logger import init_logger
from lib.cli.functions import download_data_async

np.warnings.filterwarnings('ignore')
trader_cli = RLTraderCLI()
args = trader_cli.get_args()


def run_optimize(params):
    trader_args, logger = params

    trader = RLTrader(**vars(trader_args), logger=logger)
    trader.optimize(trader_args.trials)


def optimize_concurrent(trader_args, logger):
    n_processes = trader_args.parallel_jobs

    opt_pool = ThreadPool(processes=n_processes)
    opt_pool.map(run_optimize, [((trader_args, logger)) for _ in range(n_processes)])


if __name__ == '__main__':
    logger = init_logger(__name__, show_debug=args.debug)

    if args.command == 'optimize':
        optimize_concurrent(args, logger)

    trader = RLTrader(**vars(args), logger=logger)

    if args.command == 'train':
        trader.train(n_epochs=args.epochs)
    elif args.command == 'test':
        trader.test(model_epoch=args.model_epoch, should_render=args.no_render)
    elif args.command == 'update-static-data':
        download_data_async()
