import argparse
import os
import multiprocessing
from configparser import SafeConfigParser


class RLTraderCLI:
    def __init__(self):
        config_parser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument("-f", "--from-config", help="Specify config file", metavar="FILE")
        args, remaining_argv = config_parser.parse_known_args()
        defaults = {}

        if args.from_config:
            config = SafeConfigParser()
            config.read([args.from_config])
            defaults = dict(config.items("Defaults"))

        formatter = argparse.ArgumentDefaultsHelpFormatter
        self.parser = argparse.ArgumentParser(
            formatter_class=formatter,
            parents=[config_parser],
            description=__doc__
        )

        self.parser.add_argument("--data-provider", "-o", type=str, default="static")
        self.parser.add_argument("--input-data-path", "-t", type=str, default="data/input/coinbase-1h-btc-usd.csv")
        self.parser.add_argument("--pair", "-p", type=str, default="BTC/USD")
        self.parser.add_argument("--debug", "-n", action='store_false')
        self.parser.add_argument('--mini-batches', type=int, default=1, help='Mini batches', dest='n_minibatches')
        self.parser.add_argument('--train-split-percentage', type=int, default=0.8, help='Train set percentage')
        self.parser.add_argument('--verbose-model', type=int, default=1, help='Verbose model')
        self.parser.add_argument('--params-db-path', type=str, default='sqlite:///data/params.db',
                                 help='Params path')
        self.parser.add_argument(
            '--tensor-board-path',
            type=str,
            default=os.path.join('data', 'tensorboard'),
            help='Tensorboard path',
            dest='tensorboard_path'
        )

        subparsers = self.parser.add_subparsers(help='Command', dest="command")

        opt_train_test_parser = subparsers.add_parser('opt-train-test', description='Optimize train and test')
        opt_train_test_parser.add_argument('--parallel-jobs', type=int, default=1, help='How many jobs in parallel')
        opt_train_test_parser.add_argument('--trials', type=int, default=20, help='Number of trials')
        opt_train_test_parser.add_argument('--train-epochs', type=int, default=10, help='Train for how many epochs')
        opt_train_test_parser.add_argument('--no-render', action='store_false', help='Should render the model')
        opt_train_test_parser.add_argument('--no-test', action='store_false', help='Should test the model')
        opt_train_test_parser.add_argument('--proc-number', type=int, default=multiprocessing.cpu_count(),
                                           help='How many processes to spawn')

        optimize_parser = subparsers.add_parser('optimize', description='Optimize model parameters')
        optimize_parser.add_argument('--trials', type=int, default=1, help='Number of trials')
        optimize_parser.add_argument('--parallel-jobs', type=int, default=1, help='How many jobs in parallel')
        optimize_parser.add_argument('--proc-number', type=int, default=multiprocessing.cpu_count(),
                                     help='How many processes to spawn')

        optimize_parser.add_argument('--verbose-model', type=int, default=1, help='Verbose model', dest='model_verbose')

        train_parser = subparsers.add_parser('train', description='Train model')
        train_parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')

        test_parser = subparsers.add_parser('test', description='Test model')
        test_parser.add_argument('--model-epoch', type=int, default=1, help='Model epoch index')
        test_parser.add_argument('--no-render', action='store_false', help='Do not render test')
        self.parser.set_defaults(**defaults)

    def get_args(self):
        return self.parser.parse_args()

    def get_parser(self):
        return self.parser
