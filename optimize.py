'''

A large part of the code in this file was sourced from the rl-baselines-zoo library on GitHub.
In particular, the library provides a great parameter optimization set for the PPO2 algorithm,
as well as a great example implementation using optuna.

Source: https://github.com/araffin/rl-baselines-zoo/blob/master/utils/hyperparams_opt.py

'''

import optuna

import pandas as pd
import numpy as np

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import ACKTR, PPO2

from env.BitcoinTradingEnv import BitcoinTradingEnv

# number of parallel jobs
n_jobs = 4
# maximum number of trials for finding the best hyperparams
n_trials = 20
# maximum number of timesteps per trial
n_timesteps = 200
# number of test episodes per trial
n_test_episodes = 1
# number of time steps to run before evaluating for pruning
evaluation_interval = 2

# optuna.logging.set_verbosity(optuna.logging.ERROR)


def optimize_envs(trial):
    params = {
        'n_forecasts': int(trial.suggest_loguniform('n_forecasts', 4, 100)),
        'confidence_interval': trial.suggest_uniform('confidence_interval', 0.7, 0.99),
    }

    df = pd.read_csv('./data/coinbase_hourly.csv')
    df = df.drop(['Symbol'], axis=1)

    test_len = int(len(df) * 0.2)
    train_len = int(len(df)) - test_len

    train_df = df[:train_len]
    test_df = df[train_len:]

    train_env = DummyVecEnv([lambda: BitcoinTradingEnv(
        train_df, reward_func='profit', **params)])
    test_env = DummyVecEnv([lambda: BitcoinTradingEnv(
        test_df, reward_func='profit', **params)])

    return train_env, test_env


def optimize_acktr(trial):
    params = {
        'n_steps': int(trial.suggest_loguniform('n_steps', 2, 256)),
        'gamma': trial.suggest_uniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'lr_schedule': trial.suggest_categorical('lr_schedule', ['linear', 'constant',  'double_linear_con', 'middle_drop', 'double_middle_drop']),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'vf_coef': trial.suggest_uniform('vf_coef', 0., 1.)
    }

    return params


def optimize_ppo2(trial):
    params = {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }

    return params


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

    if model.num_timesteps - model.last_time_evaluated < evaluation_interval:
        return True

    model.last_time_evaluated = model.num_timesteps

    rewards = []
    n_episodes, reward_sum = 0, 0.0

    obs = model.test_env.reset()
    while n_episodes < n_test_episodes:
        action, _ = model.predict(obs)
        obs, reward, done, _ = model.test_env.step(action)
        reward_sum += reward

        if done:
            rewards.append(reward_sum)
            reward_sum = 0.0
            n_episodes += 1
            obs = model.test_env.reset()

    mean_reward = np.mean(rewards)
    model.last_mean_test_reward = mean_reward
    model.eval_idx += 1

    model.trial.report(-1 * mean_reward, model.eval_idx)

    if model.trial.should_prune(model.eval_idx):
        model.is_pruned = True
        return False

    return True


def optimize_agent(trial):
    agent = PPO2
    policy = MlpLstmPolicy
    train_env, test_env = optimize_envs(trial)

    if agent == ACKTR:
        params = optimize_acktr(trial)
        model = ACKTR(policy, train_env, verbose=1,
                      tensorboard_log="./tensorboard", **params)
    elif agent == PPO2:
        params = optimize_ppo2(trial)
        model = PPO2(policy, train_env, verbose=1, nminibatches=1,
                     tensorboard_log="./tensorboard", **params)

    model.test_env = test_env
    model.trial = trial

    try:
        model.learn(n_timesteps, callback=learn_callback)
        model.env.close()
        test_env.close()
    except AssertionError:
        # Sometimes, random hyperparams can generate NaN
        model.env.close()
        model.test_env.close()
        raise

    is_pruned = False
    cost = np.inf

    if hasattr(model, 'is_pruned'):
        is_pruned = model.is_pruned  # pylint: disable=no-member
        cost = -1 * model.last_mean_test_reward  # pylint: disable=no-member

    del model.env, model.test_env
    del model

    if is_pruned:
        raise optuna.structs.TrialPruned()

    return cost


def optimize():
    study = optuna.create_study(
        study_name='optimize_profit', storage='sqlite:///agents.db', load_if_exists=True)

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
