import gym
import optuna
import pandas as pd
import numpy as np

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2

from pathlib import Path

from env.BitcoinTradingEnv import BitcoinTradingEnv
from util.indicators import add_indicators


curr_idx = -1
reward_strategy = 'sortino'
input_data_file = 'data/coinbase_hourly.csv'
params_db_file = 'sqlite:///params.db'

study_name = 'ppo2_' + reward_strategy
study = optuna.load_study(study_name=study_name, storage=params_db_file)
params = study.best_trial.params

print("Training PPO2 agent with params:", params)
print("Best trial reward:", -1 * study.best_trial.value)

df = pd.read_csv(input_data_file)
df = df.drop(['Symbol'], axis=1)
df = df.sort_values(['Date'])
df = add_indicators(df.reset_index())

test_len = int(len(df) * 0.2)
train_len = int(len(df)) - test_len

train_df = df[:train_len]
test_df = df[train_len:]

train_env = DummyVecEnv([lambda: BitcoinTradingEnv(
    train_df, reward_func=reward_strategy, forecast_len=int(params['forecast_len']), confidence_interval=params['confidence_interval'])])

test_env = DummyVecEnv([lambda: BitcoinTradingEnv(
    test_df, reward_func=reward_strategy, forecast_len=int(params['forecast_len']), confidence_interval=params['confidence_interval'])])

model_params = {
    'n_steps': int(params['n_steps']),
    'gamma': params['gamma'],
    'learning_rate': params['learning_rate'],
    'ent_coef': params['ent_coef'],
    'cliprange': params['cliprange'],
    'noptepochs': int(params['noptepochs']),
    'lam': params['lam'],
}

if curr_idx == -1:
    model = PPO2(MlpLnLstmPolicy, train_env, verbose=0, nminibatches=1,
            tensorboard_log=Path("./tensorboard").name, **model_params)
else:
    model = PPO2.load('./agents/ppo2_' + reward_strategy + '_' + str(curr_idx) + '.pkl', env=train_env)

for idx in range(curr_idx + 1, 10):
    print('[', idx, '] Training for: ', train_len, ' time steps')

    model.learn(total_timesteps=train_len)

    obs = test_env.reset()
    done, reward_sum = False, 0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        reward_sum += reward

    print('[', idx, '] Total reward: ', reward_sum, ' (' + reward_strategy + ')')
    model.save('./agents/ppo2_' + reward_strategy + '_' + str(idx) + '.pkl')
