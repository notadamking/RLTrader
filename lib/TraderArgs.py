import argparse
import os

class TraderArgs:
    def __init__(self):
        formatter = argparse.ArgumentDefaultsHelpFormatter
        self.parser = argparse.ArgumentParser(description='Trainer', formatter_class=formatter)

        self.parser.add_argument("--data-provider", "-d", type=str, default="static")
        self.parser.add_argument("--exchange", "-e", type=str, default="coinbase")
        self.parser.add_argument("--pair", "-p", type=str, default="BTC/USD")

        subparsers = self.parser.add_subparsers(help='Command', dest="command")

        optimize_parser = subparsers.add_parser('optimize', description='Optimize model parameters')
        optimize_parser.add_argument('--trials', type=int, default=1, help='Number of trials')
        optimize_parser.add_argument('--reward-strategy', type=str, default='sortino', help='Rewarding strategy')
        optimize_parser.add_argument(
            '--tensor-board-path',
            type=str,
            default=os.path.join('data', 'tensorboard'),
            help='Tensorboard path',
            dest='tensorboard_path'
        )
        optimize_parser.add_argument('--params-db-path', type=str, default='sqlite:///data/params.db',
                                     help='Params path')
        optimize_parser.add_argument('--verbose-model', type=int, default=1, help='Verbose model', dest='model_verbose')
        optimize_parser.add_argument('--mini-batches', type=int, default=1, help='Mini batches', dest='nminibatches')
        optimize_parser.add_argument('--validation-set-percentage', type=int, default=0.8,
                                     help='Validation set percentage')
        optimize_parser.add_argument('--test-set-percentage', type=int, default=0.8, help='Test set percentage')

        train_parser = subparsers.add_parser('train', description='Train model')
        train_parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')

        test_parser = subparsers.add_parser('test', description='Test model')
        test_parser.add_argument('--model-epoch', type=int, default=1, help='Model epoch index')
        test_parser.add_argument('--no-render', action='store_true', help='Do not render test')

    def get_args(self):
        return self.parser.parse_args()

    def get_parser(self):
        return self.parser