import numpy as np

from multiprocessing import Process

from lib.cli.RLTraderCLI import RLTraderCLI
from lib.util.logger import init_logger
from lib.cli.functions import download_data_async
from lib.env.reward import BaseRewardStrategy, IncrementalProfit, WeightedUnrealizedProfit

np.warnings.filterwarnings('ignore')

trader_cli = RLTraderCLI()
args = trader_cli.get_args()

rewards = {"incremental-profit": IncrementalProfit, "weighted-unrealized-profit": WeightedUnrealizedProfit}
reward_strategy = rewards[args.reward_strat]


def run_optimize(args, logger):
    from lib.RLTrader import RLTrader

    trader = RLTrader(**vars(args), logger=logger, reward_strategy=reward_strategy)
    trader.optimize(n_trials=args.trials)


if __name__ == '__main__':
    logger = init_logger(__name__, show_debug=args.debug)

    if args.command == 'optimize':
        n_processes = args.parallel_jobs

        processes = []
        for _ in range(n_processes):
            processes.append(Process(target=run_optimize, args=(args, logger)))

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

    from lib.RLTrader import RLTrader

    trader = RLTrader(**vars(args), logger=logger, reward_strategy=reward_strategy)

    if args.command == 'train':
        trader.train(n_epochs=args.epochs,
                     save_every=args.save_every,
                     test_trained_model=args.test_trained,
                     render_test_env=args.render_test,
                     render_report=args.render_report,
                     save_report=args.save_report)
    elif args.command == 'test':
        trader.test(model_epoch=args.model_epoch,
                    render_env=args.render_env,
                    render_report=args.render_report,
                    save_report=args.save_report)
    elif args.command == 'update-static-data':
        download_data_async()
