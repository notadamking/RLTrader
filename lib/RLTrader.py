import optuna
import pandas as pd
import numpy as np

from os import path
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.policies import BasePolicy, MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

from lib.env.TradingEnv import TradingEnv
from lib.data.providers.dates import ProviderDateFormat
from lib.data.providers import StaticDataProvider
from lib.data.features.indicators import add_indicators
from lib.util.logger import init_logger


class RLTrader:
    def __init__(self, modelClass: BaseRLModel = PPO2, policyClass: BasePolicy = MlpPolicy, **kwargs):
        self.logger = init_logger(__name__, show_debug=kwargs.get('show_debug', True))

        self.Model = modelClass
        self.Policy = policyClass
        self.tensorboard_path = kwargs.get('tensorboard_path', None)
        self.input_data_path = kwargs.get('input_data_path', None)
        self.params_db_path = kwargs.get('params_db_path', 'sqlite:///data/params.db')

        self.date_format = kwargs.get('date_format', ProviderDateFormat.DATETIME_HOUR_12)

        self.model_verbose = kwargs.get('model_verbose', 1)
        self.nminibatches = kwargs.get('nminibatches', 1)
        self.train_split_percentage = kwargs.get('train_split_percentage', 0.8)

        self.initialize_data()
        self.initialize_optuna()

        self.logger.debug(f'Initialize RLTrader: {self.study_name}')

    def initialize_data(self):
        if self.input_data_path is None:
            self.input_data_path = path.join('data', 'input', 'coinbase_hourly.csv')

        data_columns = {'Date': 'Date', 'Open': 'Open', 'High': 'High',
                        'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume BTC'}

        self.data_provider = StaticDataProvider(date_format=self.date_format,
                                                csv_data_path=self.input_data_path,
                                                data_columns=data_columns)

        self.logger.debug(f'Initialized Features: {self.data_provider.columns}')

    def initialize_optuna(self):
        try:
            train_env = DummyVecEnv([lambda: TradingEnv(self.data_provider)])
            model = self.Model(self.Policy, train_env, nminibatches=1)
            self.study_name = f'{model.__class__.__name__}__{model.act_model.__class__.__name__}'
        except:
            self.study_name = f'UnknownModel__UnknownPolicy'

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
        train_provider, test_provider = self.data_provider.split_provider_train_test(self.train_split_percentage)
        train_provider, validation_provider = train_provider.split_provider_train_test(self.train_split_percentage)

        del test_provider

        train_env = SubprocVecEnv([lambda: TradingEnv(train_provider, i) for i in range(2)])
        validation_env = SubprocVecEnv([lambda: TradingEnv(validation_provider, i) for i in range(2)])

        model_params = self.optimize_agent_params(trial)
        model = self.Model(self.Policy, train_env, verbose=self.model_verbose, nminibatches=self.nminibatches,
                           tensorboard_log=self.tensorboard_path, **model_params)

        last_reward = -np.finfo(np.float16).max
        n_steps_per_eval = int(len(train_provider.data_frame) / n_prune_evals_per_trial)

        for eval_idx in range(n_prune_evals_per_trial):
            try:
                model.learn(n_steps_per_eval)
            except AssertionError:
                raise

            rewards = []
            n_episodes, reward_sum = 0, 0.0

            state = None
            obs = validation_env.reset()
            while n_episodes < n_tests_per_eval:
                action, state = model.predict(obs, state=state)
                obs, reward, done, _ = validation_env.step(action)
                reward_sum += reward

                if done:
                    rewards.append(reward_sum)
                    reward_sum = 0.0
                    n_episodes += 1
                    obs = validation_env.reset()

            last_reward = np.mean(rewards)
            trial.report(-1 * last_reward, eval_idx)

            if trial.should_prune(eval_idx):
                raise optuna.structs.TrialPruned()

        return -1 * last_reward

    def optimize(self, n_trials: int = 1, n_parallel_jobs: int = 1, *optimize_params):
        try:
            self.optuna_study.optimize(
                self.optimize_params, n_trials=n_trials, n_jobs=n_parallel_jobs, *optimize_params)
        except KeyboardInterrupt:
            pass

        self.logger.info(f'Finished trials: {len(self.optuna_study.trials)}')

        self.logger.info(f'Best trial: {self.optuna_study.best_trial.value}')

        self.logger.info('Params: ')
        for key, value in self.optuna_study.best_trial.params.items():
            self.logger.info(f'    {key}: {value}')

        return self.optuna_study.trials_dataframe()

    def train(self, n_epochs: int = 10, steps_per_epoch: int = 1000, test_trained_model: bool = False, render_trained_model: bool = False):
        train_provider, test_provider = self.data_provider.split_provider_train_test(self.train_split_percentage)

        del test_provider

        train_env = SubprocVecEnv([lambda: TradingEnv(train_provider, i) for i in range(4)])

        model_params = self.get_model_params()

        model = self.Model(self.Policy, train_env, verbose=self.model_verbose, nminibatches=self.nminibatches,
                           tensorboard_log=self.tensorboard_path, **model_params)

        self.logger.info(f'Training for {n_epochs} epochs')

        for model_epoch in range(0, n_epochs):
            self.logger.info(f'[{model_epoch}] Training for: {steps_per_epoch} time steps')

            model.learn(total_timesteps=steps_per_epoch)

            model_path = path.join('data', 'agents', f'{self.study_name}__{model_epoch}.pkl')
            model.save(model_path)

            #if test_trained_model:
             #   self.test(model_epoch, should_render=render_trained_model)

        self.logger.info(f'Trained {n_epochs} models')

    def test(self, model_epoch: int = 0, should_render: bool = False):
        train_provider, test_provider = self.data_provider.split_provider_train_test(self.train_split_percentage)

        del train_provider

        test_env = SubprocVecEnv([lambda: TradingEnv(test_provider, i) for i in range(4)])

        model_path = path.join('data', 'agents', f'{self.study_name}__{model_epoch}.pkl')
        model = self.Model.load(model_path, env=test_env)

        self.logger.info(f'Testing model ({self.study_name}__{model_epoch})')

        state = None
        obs, done, rewards = test_env.reset(), False, []
        while not done:
            action, state = model.predict(obs, state=state)
            obs, reward, done, _ = test_env.step(action)

            rewards.append(reward)

           # if should_render:
            #    test_env.render(mode='human')

        self.logger.info(
            f'Finished testing model ({self.study_name}__{model_epoch}): ${"{:.2f}".format(np.mean(rewards))}')
