import numpy as np

from lib.RLTrader import RLTrader
from lib.TraderArgs import TraderArgs
from lib.util.logger import init_logger

np.warnings.filterwarnings('ignore')
option_parser = TraderArgs()
args = option_parser.get_args()

if __name__ == '__main__':
    # TODO: do not inject the args from parent parser :/
    debug = False if args.no_debug else True
    logger = init_logger(__name__, show_debug=debug)

    trader = RLTrader(**vars(args), logger=logger)

    if args.command == 'optimize':
        trader.optimize(n_trials=args.trials)
    elif args.command == 'train':
        trader.train(n_epochs=args.epochs)
    elif args.command == 'test':
        trader.test(model_epoch=args.model_epoch, should_render=args.no_render)
