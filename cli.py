import numpy as np

from lib.RLTrader import RLTrader
from lib.cli.RLTraderCLI import RLTraderCLI
from lib.util.logger import init_logger

np.warnings.filterwarnings('ignore')
trader_cli = RLTraderCLI()
args = trader_cli.get_args()

if __name__ == '__main__':
    logger = init_logger(__name__, show_debug=args.debug)
    trader = RLTrader(**vars(args), logger=logger)

    if args.command == 'optimize':
        trader.optimize(n_trials=args.trials, n_parallel_jobs=args.parallel_jobs)
    elif args.command == 'train':
        trader.train(n_epochs=args.epochs)
    elif args.command == 'test':
        trader.test(model_epoch=args.model_epoch, should_render=args.no_render)
    elif args.command == 'opt-train-test':
        trader.optimize(args.trials, args.parallel_jobs)
        trader.train(n_epochs=args.train_epochs, test_trained_model=args.no_test, render_trained_model=args.no_render)
