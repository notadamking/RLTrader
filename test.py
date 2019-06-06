import os
import gym
import optuna
import pandas as pd

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2

from env.BitcoinTradingEnv import BitcoinTradingEnv
from util.indicators import add_indicators

curr_idx = 0
reward_strategy = 'sortino'
input_data_file = os.path.join('data', 'coinbase_hourly.csv')
params_db_file = 'sqlite:///params.db'

study_name = 'ppo2' + reward_strategy
study = optuna.load_study(study_name=study_name, storage=params_db_file)
params = study.best_trial.params

print("Testing PPO2 agent with params:", params)
print("Best trial:", -1 * study.best_trial.value)

df = pd.read_csv(input_data_file)
df = df.drop(['Symbol'], axis=1)
df = df.sort_values(['Date'])
df = add_indicators(df.reset_index())

test_len = int(len(df) * 0.2)
train_len = int(len(df)) - test_len

test_df = df[train_len:]

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

model = PPO2.load(os.path.join('.', 'agents', 'ppo2_' + reward_strategy + '_' + str(curr_idx) + '.pkl'), env=test_env)

obs, done = test_env.reset(), False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = test_env.step(action)

    test_env.render(mode="human")
