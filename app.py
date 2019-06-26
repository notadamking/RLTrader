import numpy as np
import argparse
from lib.data_feed.CcxtDataProvider import CcxtDataProvider
from lib.data_feed.StaticDataProvider import StaticDataProvider
from lib.RLTrader import RLTrader

np.warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Trainer')
parser.add_argument("--data-provider", "-d", type=str, default="static")
parser.add_argument("--exchange", "-e", type=str, default="coinbase")
parser.add_argument("--pair", "-p", type=str, default="BTC/USD")
subparsers = parser.add_subparsers(help='Command', dest="command")

optimize_parser = subparsers.add_parser('optimize', description='Optimize model parameters')
optimize_parser.add_argument('--trials', type=int, default=1, help='Number of trials')

train_parser = subparsers.add_parser('train', description='Train model')
train_parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')

test_parser = subparsers.add_parser('test', description='Test model')
test_parser.add_argument('--model-epoch', type=int, default=1, help='Model epoch index')
test_parser.add_argument('--no-render', action='store_true', help='Do not render test')

args = parser.parse_args()

if __name__ == '__main__':
    provider_args = {'exchange': args.exchange, 'timeframe': 1, 'unit': 'h', 'symbol_pair': args.pair}
    data_provider = StaticDataProvider(**provider_args) if args.data_provider == 'static' else CcxtDataProvider(
        **provider_args)

    trader = RLTrader(data_provider)

    if args.command == 'optimize':
        trader.optimize(n_trials=args.trials)
    elif args.command == 'train':
        trader.train(n_epochs=args.epochs)
    elif args.command == 'test':
        trader.test(model_epoch=args.model_epoch, should_render=args.no_render)
