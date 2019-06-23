'''

A large part of the code in this file was sourced from the rl-baselines-zoo library on GitHub.
In particular, the library provides a great parameter optimization set for the PPO2 algorithm,
as well as a great example implementation using optuna.

Source: https://github.com/araffin/rl-baselines-zoo/blob/master/utils/hyperparams_opt.py

'''

import optuna

import os
import pandas as pd
import numpy as np

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from pathlib import Path

from env.BitcoinTradingEnv import BitcoinTradingEnv
from util.indicators import add_indicators
from util.log import init_logger


class Optimize:
    def __init__(self):
        self.reward_strategy = 'sortino'
        self.input_data_file = os.path.join('data', 'coinbase_hourly.csv')
        self.params_db_file = 'sqlite:///params.db'

        # number of parallel jobs
        self.n_jobs = 4
        # maximum number of trials for finding the best hyperparams
        self.n_trials = 1000
        # number of test episodes per trial
        self.n_test_episodes = 3
        # number of evaluations for pruning per trial
        self.n_evaluations = 4

        self.train_df = None
        self.test_df = None

        self.logger = init_logger(__name__, testing_mode=True)

        self.logger.debug("Initialized Optimizer")

    def prepare_data(self):
        df = pd.read_csv(self.input_data_file)
        df = df.drop(['Symbol'], axis=1)
        df = df.sort_values(['Date'])
        df = add_indicators(df.reset_index())

        train_len = int(len(df) * 0.8)

        df = df[:train_len]

        validation_len = int(train_len * 0.8)
        self.train_df = df[:validation_len]
        self.test_df = df[validation_len:]

    def optimize_envs(self, trial):
        return {
            'reward_func': self.reward_strategy,
            'forecast_len': int(trial.suggest_loguniform('forecast_len', 1, 200)),
            'confidence_interval': trial.suggest_uniform('confidence_interval', 0.7, 0.99),
        }

    def optimize_ppo2(self, trial):
        return {
            'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
            'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
            'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
            'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
            'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
            'lam': trial.suggest_uniform('lam', 0.8, 1.)
        }

    def optimize_agent(self, trial):
        env_params = self.optimize_envs(trial)
        train_env = DummyVecEnv(
            [lambda: BitcoinTradingEnv(self.train_df,  **env_params)])
        test_env = DummyVecEnv(
            [lambda: BitcoinTradingEnv(self.test_df, **env_params)])

        model_params = self.optimize_ppo2(trial)
        model = PPO2(MlpLnLstmPolicy, train_env, verbose=0, nminibatches=1,
                     tensorboard_log=os.path.join('.', 'tensorboard'), **model_params)

        last_reward = -np.finfo(np.float16).max
        evaluation_interval = int(len(self.train_df) / self.n_evaluations)

        for eval_idx in range(self.n_evaluations):
            try:
                model.learn(evaluation_interval)
            except AssertionError:
                raise

            rewards = []
            n_episodes, reward_sum = 0, 0.0

            obs = test_env.reset()
            while n_episodes < self.n_test_episodes:
                action, _ = model.predict(obs)
                obs, reward, done, _ = test_env.step(action)
                reward_sum += reward

                if done:
                    rewards.append(reward_sum)
                    reward_sum = 0.0
                    n_episodes += 1
                    obs = test_env.reset()

            last_reward = np.mean(rewards)
            trial.report(-1 * last_reward, eval_idx)

            if trial.should_prune(eval_idx):
                raise optuna.structs.TrialPruned()

        return -1 * last_reward

    def log_parameters(self):
        self.logger.debug("Reward Strategy: %s" % self.reward_strategy)
        self.logger.debug("Input Data File: %s" % self.input_data_file)
        self.logger.debug("Params DB File: %s" % self.params_db_file)
        self.logger.debug("Parallel jobs: %d" % self.n_jobs)
        self.logger.debug("Trials: %d" % self.n_trials)
        self.logger.debug("Test episodes (per trial): %d" %
                          self.n_test_episodes)
        self.logger.debug("Evaluations (per trial): %d" % self.n_evaluations)
        self.logger.debug("Train DF Length: %d" % len(self.train_df))
        self.logger.debug("Test DF Length: %d" % len(self.test_df))
        self.logger.debug(
            "Features: %s", self.train_df.columns.str.cat(sep=", "))

    def optimize(self):
        if not self.train_df:
            self.logger.info("Running built-in data preparation")
            self.prepare_data()
        else:
            self.logger.info("Using provided data (Length: %d)" %
                             len(self.train_df))

        self.log_parameters()

        study_name = 'ppo2_' + self.reward_strategy
        study = optuna.create_study(
            study_name=study_name, storage=self.params_db_file, load_if_exists=True)

        try:
            study.optimize(self.optimize_agent,
                           n_trials=self.n_trials, n_jobs=self.n_jobs)
        except KeyboardInterrupt:
            pass

        self.logger.info(
            'Number of finished trials: {}'.format(len(study.trials)))

        self.logger.info('Best trial:')
        trial = study.best_trial

        self.logger.info('Value: {}'.format(trial.value))

        self.logger.info('Params: ')
        for key, value in trial.params.items():
            self.logger.info('    {}: {}'.format(key, value))

        return study.trials_dataframe()

    def model_params(self, params):
        return {
            'n_steps': int(params['n_steps']),
            'gamma': params['gamma'],
            'learning_rate': params['learning_rate'],
            'ent_coef': params['ent_coef'],
            'cliprange': params['cliprange'],
            'noptepochs': int(params['noptepochs']),
            'lam': params['lam'],
        }

    def train(self):
        if not self.train_df:
            self.logger.info("Running built-in data preparation")
            self.prepare_data()
        else:
            self.logger.info("Using provided data (Length: %d)" %
                             len(self.train_df))

        study_name = 'ppo2_' + self.reward_strategy

        study = optuna.load_study(
            study_name=study_name, storage=self.params_db_file)
        params = study.best_trial.params

        train_env = DummyVecEnv([lambda: BitcoinTradingEnv(
            self.train_df, reward_func=self.reward_strategy, forecast_len=int(params['forecast_len']), confidence_interval=params['confidence_interval'])])

        test_env = DummyVecEnv([lambda: BitcoinTradingEnv(
            self.test_df, reward_func=self.reward_strategy, forecast_len=int(params['forecast_len']), confidence_interval=params['confidence_interval'])])

        model_params = self.model_params(params)

        model = PPO2(MlpLnLstmPolicy, train_env, verbose=0, nminibatches=1,
                     tensorboard_log=os.path.join('.', 'tensorboard'), **model_params)

        models_to_train = 1
        self.logger.info("Training {} model instances".format(models_to_train))

        for idx in range(0, models_to_train):  # Not sure why we are doing this, tbh
            self.logger.info(
                f'[{idx}] Training for: {len(self.train_df)} time steps')

            model.learn(total_timesteps=len(self.train_df))

            obs = test_env.reset()
            done, reward_sum = False, 0

            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = test_env.step(action)
                reward_sum += reward

            self.logger.info(
                f'[{idx}] Total reward: {reward_sum} ({self.reward_strategy})')

            model.save(os.path.join('.', 'agents', 'ppo2_' +
                                    self.reward_strategy + '_' + str(idx) + '.pkl'))

        self.logger.info("Trained {} model instances".format(models_to_train))

    def test(self, model_instance: 0):

        study_name = 'ppo2_' + self.reward_strategy
        study = optuna.load_study(
            study_name=study_name, storage=self.params_db_file)
        params = study.best_trial.params

        test_env = DummyVecEnv([lambda: BitcoinTradingEnv(
            self.test_df, reward_func=self.reward_strategy, forecast_len=int(params['forecast_len']), confidence_interval=params['confidence_interval'])])

        model = PPO2.load(os.path.join('.', 'agents', 'ppo2_' +
                                       self.reward_strategy + '_' + str(model_instance) + '.pkl'), env=test_env)

        obs, done = test_env.reset(), False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = test_env.step(action)

            test_env.render(mode="human")


if __name__ == '__main__':
    optimizer = Optimize()
    test_mode = "FAST"  # I'm hard-coding this for now
    if test_mode == "FAST":
        optimizer.input_data_file = os.path.join('data', 'coinbase_daily.csv')
        optimizer.n_jobs = 1
        optimizer.n_trials = 1
        optimizer.n_test_episodes = 1
        optimizer.n_evaluations = 1
    # optimizer.optimize()
    optimizer.train()
    # optimizer.test()
