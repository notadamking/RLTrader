import numpy as np

from multiprocessing import Process

from lib.cli.RLTraderCLI import RLTraderCLI
from lib.util.logger import init_logger
from lib.cli.functions import download_data_async

np.warnings.filterwarnings('ignore')
trader_cli = RLTraderCLI()
args = trader_cli.get_args()


def run_optimize(args, logger):
    from lib.RLTrader import RLTrader

    trader = RLTrader(**vars(args), logger=logger)
    trader.optimize(args.trials)


if __name__ == '__main__':
    logger = init_logger(__name__, show_debug=args.debug)

    if args.command == 'optimize':
        n_processes = args.parallel_jobs

        processes = []
        for _ in range(n_processes):
            processes.append(Process(target=run_optimize, args=(args, logger)))

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

    from lib.RLTrader import RLTrader

    trader = RLTrader(**vars(args), logger=logger)

    if args.command == 'train':
        trader.train(n_epochs=args.epochs)
    elif args.command == 'test':
        trader.test(model_epoch=args.model_epoch, should_render=args.no_render, render_tearsheet=args.no_tearsheet)
    elif args.command == 'update-static-data':
        download_data_async()
