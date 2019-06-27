import optuna
import numpy as np

from os import path
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.policies import BasePolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from lib.env.BitcoinTradingEnv import BitcoinTradingEnv
from lib.util.log import init_logger
from lib.data_feed import IDataProvider


class RLTrader:
    feature_df = None
    validation_set_percentage = 0.8
    test_set_percentage = 0.8
    optuna_study = None
    study_name = None

    def __init__(self, provider: IDataProvider, model: BaseRLModel = PPO2, policy: BasePolicy = MlpLnLstmPolicy,
                 **kwargs):
        '''
        :param data_provider:
        :param model:
        :param policy:
        :param kwargs:
        '''
        self.logger = init_logger(__name__, show_debug=kwargs.get('show_debug', True))

        self.data_feed = provider
        self.model = model
        self.policy = policy
        self.reward_strategy = kwargs.get('reward_strategy', 'sortino')
        self.tensorboard_path = kwargs.get('tensorboard_path', path.join('data', 'tensorboard'))
        self.params_db_path = kwargs.get('params_db_path', 'sqlite:///data/params.db')

        self.model_verbose = kwargs.get('model_verbose', 1)
        self.nminibatches = kwargs.get('nminibatches', 1)

        self.initialize_data(
            kwargs.get('validation_set_percentage', 0.8),
            kwargs.get('test_set_percentage', 0.8)
        )

        self.logger.debug(f'Reward Strategy: {self.reward_strategy}')

    def initialize_data(self, validation_set_percentage: float, test_set_percentage: float):
        self.feature_df = self.data_feed.get_data()

        self.validation_set_percentage = validation_set_percentage
        self.test_set_percentage = test_set_percentage

        self.logger.debug(f'Initialized Features: {self.feature_df.columns.str.cat(sep=", ")}')

    def initialize_optuna(self, should_create: bool = False):
        self.study_name = f'{self.model.__class__.__name__}__{self.policy.__class__.__name__}__{self.reward_strategy}'

        if should_create:
            self.optuna_study = optuna.create_study(
                study_name=self.study_name, storage=self.params_db_path, load_if_exists=True)
        else:
            self.optuna_study = optuna.load_study(
                study_name=self.study_name, storage=self.params_db_path)

        self.logger.debug('Initialized Optuna:')

        try:
            self.logger.debug(
                f'Best reward in ({len(self.optuna_study.trials)}) trials: {-self.optuna_study.best_value}')
        except:
            self.logger.debug('No trials have been finished yet.')

    def get_env_params(self):
        params = self.optuna_study.best_trial.params
        return {
            'reward_strategy': self.reward_strategy,
            'forecast_steps': int(params['forecast_steps']),
            'forecast_alpha': params['forecast_alpha'],
        }

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

    def optimize_env_params(self, trial):
        return {
            'forecast_steps': int(trial.suggest_loguniform('forecast_steps', 1, 200)),
            'forecast_alpha': trial.suggest_uniform('forecast_alpha', 0.001, 0.30),
        }

    def optimize_agent_params(self, trial):
        if self.model != PPO2:
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

    def optimize_params(self, trial, n_prune_evals_per_trial: int = 4, n_tests_per_eval: int = 1,
                        speedup_factor: int = 10):
        env_params = self.optimize_env_params(trial)

        full_train_len = self.test_set_percentage * len(self.feature_df)
        optimize_train_len = int(
            self.validation_set_percentage * full_train_len)
        train_len = int(optimize_train_len / speedup_factor)
        train_start = optimize_train_len - train_len

        train_df = self.feature_df[train_start:optimize_train_len]
        validation_df = self.feature_df[optimize_train_len:]

        train_env = DummyVecEnv(
            [lambda: BitcoinTradingEnv(train_df, **env_params)])
        validation_env = DummyVecEnv(
            [lambda: BitcoinTradingEnv(validation_df, **env_params)])

        model_params = self.optimize_agent_params(trial)
        model = self.model(self.policy, train_env, verbose=self.model_verbose, nminibatches=self.nminibatches,
                           tensorboard_log=self.tensorboard_path, **model_params)

        last_reward = -np.finfo(np.float16).max
        evaluation_interval = int(
            train_len / n_prune_evals_per_trial)

        for eval_idx in range(n_prune_evals_per_trial):
            try:
                model.learn(evaluation_interval)
            except AssertionError:
                raise

            rewards = []
            n_episodes, reward_sum = 0, 0.0

            obs = validation_env.reset()
            while n_episodes < n_tests_per_eval:
                action, _ = model.predict(obs)
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

    def optimize(self, n_trials: int = 10, n_parallel_jobs: int = 4, *optimize_params):
        self.initialize_optuna(should_create=True)

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

    def train(self, n_epochs: int = 1, iters_per_epoch: int = 1):
        self.initialize_optuna()

        env_params = self.get_env_params()

        train_len = int(self.test_set_percentage * len(self.feature_df))
        train_df = self.feature_df[:train_len]

        train_env = DummyVecEnv(
            [lambda: BitcoinTradingEnv(train_df, **env_params)])

        model_params = self.get_model_params()

        model = self.model(self.policy, train_env, verbose=self.model_verbose, nminibatches=self.nminibatches,
                           tensorboard_log=self.tensorboard_path, **model_params)

        self.logger.info(f'Training for {n_epochs} epochs')

        n_timesteps = len(train_df) * iters_per_epoch

        for model_epoch in range(0, n_epochs):
            self.logger.info(
                f'[{model_epoch}] Training for: {n_timesteps} time steps')

            model.learn(total_timesteps=n_timesteps)

            model_path = path.join(
                'data', 'agents', f'{self.study_name}__{model_epoch}.pkl')
            model.save(model_path)

        self.logger.info(f'Trained {n_epochs} models')

    def test(self, model_epoch: int = 0, should_render: bool = True):
        self.initialize_optuna()
        env_params = self.get_env_params()

        train_len = int(self.test_set_percentage * len(self.feature_df))
        test_df = self.feature_df[train_len:]

        test_env = DummyVecEnv(
            [lambda: BitcoinTradingEnv(test_df, **env_params)])

        model_path = path.join('data', 'agents', f'{self.study_name}__{model_epoch}.pkl')
        model = self.model.load(model_path, env=test_env)

        self.logger.info(
            f'Testing model ({self.study_name}__{model_epoch})')

        obs, done, reward_sum = test_env.reset(), False, 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _ = test_env.step(action)

            reward_sum += reward

            if should_render:
                test_env.render(mode='human')

        self.logger.info(
            f'Finished testing model ({self.study_name}__{model_epoch}): ${"{:.2f}".format(str(reward_sum))}'
        )
