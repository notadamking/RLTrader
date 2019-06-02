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

from env.BitcoinTradingEnv import BitcoinTradingEnv
from util.indicators import add_indicators

# number of parallel jobs
n_jobs = 2
# maximum number of trials for finding the best hyperparams
n_trials = 1000
# number of test episodes per trial
n_test_episodes = 3
# number of evaluations for pruning per trial
n_evaluations = 4


df = pd.read_csv('./data/coinbase_hourly.csv')
df = df.drop(['Symbol'], axis=1)
df = df.sort_values(['Date'])
df = add_indicators(df.reset_index())

train_len = int(len(df)) - int(len(df) * 0.2)
train_df = df[:train_len]


def optimize_envs(trial):
    return {
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


def learn_callback(_locals, _globals):
    """
    Callback for monitoring stable-baselines learning progress.
    :param _locals: (dict)
    :param _globals: (dict)
    :return: (bool) If False: stop training
    """
    model = _locals['self']

    if not hasattr(model, 'is_pruned'):
        model.is_pruned = False
        model.last_mean_test_reward = -np.inf
        model.last_time_evaluated = 0
        model.eval_idx = 0

    if model.n_timesteps - model.last_time_evaluated < model.n_timesteps / n_evaluations or model.n_timesteps >= len(model.train_df):
        return True

    model.last_time_evaluated = model.n_timesteps

    rewards = []
    n_episodes, reward_sum = 0, 0.0

    test_df = model.train_df.copy()

    test_env = DummyVecEnv([lambda: BitcoinTradingEnv(
        test_df[model.n_timesteps:], reward_func='omega', **model.env_params)])

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

    model.last_mean_test_reward = np.mean(rewards) / len(test_df)
    model.eval_idx += 1

    model.trial.report(-1 * model.last_mean_test_reward, model.eval_idx)

    if model.trial.should_prune(model.eval_idx):
        model.is_pruned = True
        return False

    return True


def optimize_agent(trial):
    env_params = optimize_envs(trial)
    train_env = DummyVecEnv([lambda: BitcoinTradingEnv(
        train_df, reward_func='omega', **env_params)])

    model_params = optimize_ppo2(trial)
    model = PPO2(MlpLnLstmPolicy, train_env, verbose=1, nminibatches=1,
                     tensorboard_log="./tensorboard", **model_params)

    model.trial = trial
    model.n_timesteps = len(train_df)
    model.train_df = train_df
    model.env_params = env_params

    try:
        model.learn(model.n_timesteps, callback=learn_callback)
        model.env.close()
    except AssertionError:
        model.env.close()
        raise

    is_pruned = False
    cost = np.inf

    if hasattr(model, 'is_pruned'):
        is_pruned = model.is_pruned  # pylint: disable=no-member
        cost = -1 * model.last_mean_test_reward  # pylint: disable=no-member

    if is_pruned:
        raise optuna.structs.TrialPruned()

    return cost


def optimize():
    study = optuna.create_study(
        study_name='ppo2_omega', storage='sqlite:///params.db', load_if_exists=True)

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
