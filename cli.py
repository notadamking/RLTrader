import numpy as np
from deco import concurrent
from lib.RLTrader import RLTrader
from lib.cli.RLTraderCLI import RLTraderCLI
from lib.util.logger import init_logger
from update_data import download_async

np.warnings.filterwarnings('ignore')
trader_cli = RLTraderCLI()
args = trader_cli.get_args()


@concurrent(processes=args.proc_number)
def run_concurrent_optimize(trader: RLTrader, args):
    trader.optimize(args.trials, args.trials, args.parallel_jobs)


if __name__ == '__main__':
    logger = init_logger(__name__, show_debug=args.debug)
    trader = RLTrader(**vars(args), logger=logger)

    if args.command == 'optimize':
        run_concurrent_optimize(trader, args)
    elif args.command == 'train':
        trader.train(n_epochs=args.epochs)
    elif args.command == 'test':
        trader.test(model_epoch=args.model_epoch, should_render=args.no_render)
    elif args.command == 'opt-train-test':
        run_concurrent_optimize(trader, args)
        trader.train(
            n_epochs=args.train_epochs,
            test_trained_model=args.no_test,
            render_trained_model=args.no_render
        )
    elif args.command == 'update-static':
        download_async()