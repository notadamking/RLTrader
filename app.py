import tensorflow as tf
ncpu = 6
config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu)
tf.Session(config=config).__enter__()

import numpy as np
from lib.TraderArgs import TraderArgs
from lib.data_feed.CcxtDataProvider import CcxtDataProvider
from lib.data_feed.StaticDataProvider import StaticDataProvider
from lib.RLTrader import RLTrader

np.warnings.filterwarnings('ignore')
option_parser = TraderArgs()
args = option_parser.get_args()

if __name__ == '__main__':
    provider_args = {'exchange': args.exchange, 'timeframe': 1, 'unit': 'h', 'symbol_pair': args.pair}
    data_provider = StaticDataProvider(**provider_args) if args.data_provider == 'static' else CcxtDataProvider(
        **provider_args)

    # TODO: do not inject the args from parent parser :/
    trader = RLTrader(data_provider, **vars(args))

    if args.command == 'optimize':
        trader.optimize(n_trials=args.trials)
    elif args.command == 'train':
        trader.train(n_epochs=args.epochs)
    elif args.command == 'test':
        trader.test(model_epoch=args.model_epoch, should_render=args.no_render)
