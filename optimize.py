'''

A large part of the code in this file was sourced from the rl-baselines-zoo library on GitHub.
In particular, the library provides a great parameter optimization set for the PPO2 algorithm,
as well as a great example implementation using optuna.

Source: https://github.com/araffin/rl-baselines-zoo/blob/master/utils/hyperparams_opt.py

'''

import optuna

import pandas as pd
import numpy as np

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from pathlib import Path

from env.BitcoinTradingEnv import BitcoinTradingEnv
from util.indicators import add_indicators


reward_strategy = 'sortino'
input_data_file = 'data/coinbase_hourly.csv'
params_db_file = 'sqlite:///params.db'

# number of parallel jobs
n_jobs = 4
# maximum number of trials for finding the best hyperparams
n_trials = 1000
# number of test episodes per trial
n_test_episodes = 3
# number of evaluations for pruning per trial
n_evaluations = 4


df = pd.read_csv(input_data_file)
df = df.drop(['Symbol'], axis=1)
df = df.sort_values(['Date'])
df = add_indicators(df.reset_index())

train_len = int(len(df) * 0.8)

df = df[:train_len]

validation_len = int(train_len * 0.8)
train_df = df[:validation_len]
test_df = df[validation_len:]


def optimize_envs(trial):
    return {
        'reward_func': reward_strategy,
        'forecast_len': int(trial.suggest_loguniform('forecast_len', 1, 200)),
        'confidence_interval': trial.suggest_uniform('confidence_interval', 0.7, 0.99),
    }


def optimize_ppo2(trial):
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }


def optimize_agent(trial):
    env_params = optimize_envs(trial)
    train_env = DummyVecEnv(
        [lambda: BitcoinTradingEnv(train_df,  **env_params)])
    test_env = DummyVecEnv(
        [lambda: BitcoinTradingEnv(test_df, **env_params)])

    model_params = optimize_ppo2(trial)
    model = PPO2(MlpLnLstmPolicy, train_env, verbose=0, nminibatches=1,
                 tensorboard_log=Path("./tensorboard").name, **model_params)

    last_reward = -np.finfo(np.float16).max
    evaluation_interval = int(len(train_df) / n_evaluations)

    for eval_idx in range(n_evaluations):
        try:
            model.learn(evaluation_interval)
        except AssertionError:
            raise

        rewards = []
        n_episodes, reward_sum = 0, 0.0

        obs = test_env.reset()
        while n_episodes < n_test_episodes:
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


def optimize():
    study_name = 'ppo2_' + reward_strategy
    study = optuna.create_study(
        study_name=study_name, storage=params_db_file, load_if_exists=True)

    try:
        study.optimize(optimize_agent, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        pass

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)

    print('Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return study.trials_dataframe()


if __name__ == '__main__':
    optimize()
