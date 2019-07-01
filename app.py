import numpy as np

from lib.RLTrader import RLTrader
from lib.TraderArgs import TraderArgs

np.warnings.filterwarnings('ignore')
option_parser = TraderArgs()
args = option_parser.get_args()

if __name__ == '__main__':
    # TODO: do not inject the args from parent parser :/
    trader = RLTrader(**vars(args))

    if args.command == 'optimize':
        trader.optimize(n_trials=args.trials)
    elif args.command == 'train':
        trader.train(n_epochs=args.epochs)
    elif args.command == 'test':
        trader.test(model_epoch=args.model_epoch, should_render=args.no_render)
