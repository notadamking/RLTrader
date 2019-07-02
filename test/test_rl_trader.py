import unittest

from lib.RLTrader import RLTrader
from lib.TraderArgs import TraderArgs


class TestRLTrader(unittest.TestCase):
    def setUp(self):
        self.parser = TraderArgs().get_parser()

    def test_that_args_get_injected_correctly(self):
        args = self.parser.parse_args(['optimize'])
        sut = RLTrader(args)

        self.assertEqual(sut.tensorboard_path, args.tensorboard_path)
        self.assertEqual(sut.params_db_path, args.params_db_path)
        self.assertEqual(sut.model_verbose, args.model_verbose)
        self.assertEqual(sut.nminibatches, args.nminibatches)
