import numpy as np
import argparse
from lib.data_feed.CcxtDataProvider import CcxtDataProvider
from lib.data_feed.StaticDataProvider import StaticDataProvider
from lib.RLTrader import RLTrader

np.warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Trainer')
parser.add_argument("--data-provider", "-d", type=str, default="static")
parser.add_argument("--exchange", "-e", type=str, default="coinbase")
subparsers = parser.add_subparsers(help='Command', dest="command")

optimize_parser = subparsers.add_parser('optimize', description='Optimize model parameters')
train_parser = subparsers.add_parser('train', description='Train model')
test_parser = subparsers.add_parser('test', description='Test model')

args = parser.parse_args()

if __name__ == '__main__':
    if args.data_provider == 'static':
        data_provider = StaticDataProvider(exchange=args.exchange, timeframe='1', unit='h', symbol_pair='BTC/USD')
    else:
        data_provider = CcxtDataProvider(exchange=args.exchange, timeframe='1', unit='h', symbol_pair='BTC/USD')

    trader = RLTrader(data_provider)

    if args.command == 'optimize':
        trader.optimize(n_trials=1)
    elif args.command == 'train':
        trader.train(n_epochs=1, test_trained_model=True, render_trained_model=True)
    elif args.command == 'test':
        trader.test(model_epoch=1, should_render=True)
