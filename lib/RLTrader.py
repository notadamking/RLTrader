import os
import optuna
import numpy as np
import pandas as pd
import quantstats as qs

from os import path
from typing import Dict

from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.policies import BasePolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2

from lib.env.TradingEnv import TradingEnv
from lib.env.reward import BaseRewardStrategy, IncrementalProfit, WeightedUnrealizedProfit
from lib.data.providers.dates import ProviderDateFormat
from lib.data.providers import BaseDataProvider,  StaticDataProvider, ExchangeDataProvider
from lib.util.logger import init_logger


def make_env(data_provider: BaseDataProvider, rank: int = 0, seed: int = 0):
    def _init():
        env = TradingEnv(data_provider)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)

    return _init


class RLTrader:
    data_provider = None
    study_name = None

    def __init__(self,
                 model: BaseRLModel = PPO2,
                 policy: BasePolicy = MlpLnLstmPolicy,
                 reward_strategy: BaseRewardStrategy = IncrementalProfit,
                 exchange_args: Dict = {},
                 **kwargs):
        self.logger = kwargs.get('logger', init_logger(__name__, show_debug=kwargs.get('show_debug', True)))

        self.Model = model
        self.Policy = policy
        self.Reward_Strategy = reward_strategy
        self.exchange_args = exchange_args
        self.tensorboard_path = kwargs.get('tensorboard_path', None)
        self.input_data_path = kwargs.get('input_data_path', 'data/input/coinbase-1h-btc-usd.csv')
        self.params_db_path = kwargs.get('params_db_path', 'sqlite:///data/params.db')

        self.date_format = kwargs.get('date_format', ProviderDateFormat.DATETIME_HOUR_24)

        self.model_verbose = kwargs.get('model_verbose', 1)
        self.n_envs = kwargs.get('n_envs', os.cpu_count())
        self.n_minibatches = kwargs.get('n_minibatches', self.n_envs)
        self.train_split_percentage = kwargs.get('train_split_percentage', 0.8)
        self.data_provider = kwargs.get('data_provider', 'static')

        self.initialize_data()
        self.initialize_optuna()

        self.logger.debug(f'Initialize RLTrader: {self.study_name}')

    def initialize_data(self):
        if self.data_provider == 'static':
            if not os.path.isfile(self.input_data_path):
                class_dir = os.path.dirname(__file__)
                self.input_data_path = os.path.realpath(os.path.join(class_dir, "../{}".format(self.input_data_path)))

            data_columns = {'Date': 'Date', 'Open': 'Open', 'High': 'High',
                            'Low': 'Low', 'Close': 'Close', 'Volume': 'VolumeFrom'}

            self.data_provider = StaticDataProvider(date_format=self.date_format,
                                                    csv_data_path=self.input_data_path,
                                                    data_columns=data_columns)
        elif self.data_provider == 'exchange':
            self.data_provider = ExchangeDataProvider(**self.exchange_args)

        self.logger.debug(f'Initialized Features: {self.data_provider.columns}')

    def initialize_optuna(self):
        try:
            train_env = DummyVecEnv([lambda: TradingEnv(self.data_provider)])
            model = self.Model(self.Policy, train_env, nminibatches=1)
            strategy = self.Reward_Strategy()

            self.study_name = f'{model.__class__.__name__}__{model.act_model.__class__.__name__}__{strategy.__class__.__name__}'
        except:
            self.study_name = f'UnknownModel__UnknownPolicy__UnknownStrategy'

        self.optuna_study = optuna.create_study(
            study_name=self.study_name, storage=self.params_db_path, load_if_exists=True)

        self.logger.debug('Initialized Optuna:')

        try:
            self.logger.debug(
                f'Best reward in ({len(self.optuna_study.trials)}) trials: {self.optuna_study.best_value}')
        except:
            self.logger.debug('No trials have been finished yet.')

    def get_model_params(self):
        params = self.optuna_study.best_trial.params
        return {
            'n_steps': int(params['n_steps']),
            'gamma': params['gamma'],
            'learning_rate': params['learning_rate'],
            'ent_coef': params['ent_coef'],
            'cliprange': params['cliprange'],
            'noptepochs': int(params['noptepochs']),
            'lam': params['lam'],
        }

    def optimize_agent_params(self, trial):
        if self.Model != PPO2:
            return {'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.)}

        return {
            'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
            'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
            'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
            'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
            'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
            'lam': trial.suggest_uniform('lam', 0.8, 1.)
        }

    def optimize_params(self, trial, n_prune_evals_per_trial: int = 2, n_tests_per_eval: int = 1):
        train_provider, test_provider = self.data_provider.split_data_train_test(self.train_split_percentage)
        train_provider, validation_provider = train_provider.split_data_train_test(self.train_split_percentage)

        del test_provider

        train_env = DummyVecEnv([lambda: TradingEnv(train_provider)])
        validation_env = DummyVecEnv([lambda: TradingEnv(validation_provider)])

        model_params = self.optimize_agent_params(trial)
        model = self.Model(self.Policy,
                           train_env,
                           verbose=self.model_verbose,
                           nminibatches=1,
                           tensorboard_log=self.tensorboard_path,
                           **model_params)

        last_reward = -np.finfo(np.float16).max
        n_steps_per_eval = int(len(train_provider.data_frame) / n_prune_evals_per_trial)

        for eval_idx in range(n_prune_evals_per_trial):
            try:
                model.learn(n_steps_per_eval)
            except AssertionError:
                raise

            rewards = []
            n_episodes, reward_sum = 0, 0.0

            trades = train_env.get_attr('trades')

            if len(trades[0]) < 1:
                self.logger.info(f'Pruning trial for not making any trades: {eval_idx}')
                raise optuna.structs.TrialPruned()

            state = None
            obs = validation_env.reset()
            while n_episodes < n_tests_per_eval:
                action, state = model.predict(obs, state=state)
                obs, reward, done, _ = validation_env.step([action])

                reward_sum += reward[0]

                if all(done):
                    rewards.append(reward_sum)
                    reward_sum = 0.0
                    n_episodes += 1
                    obs = validation_env.reset()

            last_reward = np.mean(rewards)
            trial.report(-1 * last_reward, eval_idx)

            if trial.should_prune(eval_idx):
                raise optuna.structs.TrialPruned()

        return -1 * last_reward

    def optimize(self, n_trials: int = 20):
        try:
            self.optuna_study.optimize(self.optimize_params, n_trials=n_trials, n_jobs=1)
        except KeyboardInterrupt:
            pass

        self.logger.info(f'Finished trials: {len(self.optuna_study.trials)}')

        self.logger.info(f'Best trial: {self.optuna_study.best_trial.value}')

        self.logger.info('Params: ')
        for key, value in self.optuna_study.best_trial.params.items():
            self.logger.info(f'    {key}: {value}')

        return self.optuna_study.trials_dataframe()

    def train(self,
              n_epochs: int = 10,
              save_every: int = 1,
              test_trained_model: bool = True,
              render_test_env: bool = False,
              render_report: bool = True,
              save_report: bool = False):
        train_provider, test_provider = self.data_provider.split_data_train_test(self.train_split_percentage)

        del test_provider

        train_env = SubprocVecEnv([make_env(train_provider, i) for i in range(self.n_envs)])

        model_params = self.get_model_params()

        model = self.Model(self.Policy,
                           train_env,
                           verbose=self.model_verbose,
                           nminibatches=self.n_minibatches,
                           tensorboard_log=self.tensorboard_path,
                           **model_params)

        self.logger.info(f'Training for {n_epochs} epochs')

        steps_per_epoch = len(train_provider.data_frame)

        for model_epoch in range(0, n_epochs):
            self.logger.info(f'[{model_epoch}] Training for: {steps_per_epoch} time steps')

            model.learn(total_timesteps=steps_per_epoch)

            if model_epoch % save_every == 0:
                model_path = path.join('data', 'agents', f'{self.study_name}__{model_epoch}.pkl')
                model.save(model_path)

                if test_trained_model:
                    self.test(model_epoch,
                              render_env=render_test_env,
                              render_report=render_report,
                              save_report=save_report)

        self.logger.info(f'Trained {n_epochs} models')

    def test(self, model_epoch: int = 0, render_env: bool = True, render_report: bool = True, save_report: bool = False):
        train_provider, test_provider = self.data_provider.split_data_train_test(self.train_split_percentage)

        del train_provider

        init_envs = DummyVecEnv([make_env(test_provider) for _ in range(self.n_envs)])

        model_path = path.join('data', 'agents', f'{self.study_name}__{model_epoch}.pkl')
        model = self.Model.load(model_path, env=init_envs)

        test_env = DummyVecEnv([make_env(test_provider) for _ in range(1)])

        self.logger.info(f'Testing model ({self.study_name}__{model_epoch})')

        zero_completed_obs = np.zeros((self.n_envs,) + init_envs.observation_space.shape)
        zero_completed_obs[0, :] = test_env.reset()

        state = None
        rewards = []

        for _ in range(len(test_provider.data_frame)):
            action, state = model.predict(zero_completed_obs, state=state)
            obs, reward, done, info = test_env.step([action[0]])

            zero_completed_obs[0, :] = obs

            rewards.append(reward)

            if render_env:
                test_env.render(mode='human')

            if done:
                net_worths = pd.DataFrame({
                    'Date': info[0]['timestamps'],
                    'Balance': info[0]['net_worths'],
                })

                net_worths.set_index('Date', drop=True, inplace=True)
                returns = net_worths.pct_change()[1:]

                if render_report:
                    qs.plots.snapshot(returns.Balance, title='RL Trader Performance')

                if save_report:
                    reports_path = path.join('data', 'reports', f'{self.study_name}__{model_epoch}.html')
                    qs.reports.html(returns.Balance, file=reports_path)

        self.logger.info(
            f'Finished testing model ({self.study_name}__{model_epoch}): ${"{:.2f}".format(np.sum(rewards))}')
