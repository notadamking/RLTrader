import gym
import optuna
import pandas as pd
import numpy as np

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2

from env.BitcoinTradingEnv import BitcoinTradingEnv
from util.indicators import add_indicators


study = optuna.load_study(study_name='ppo2_calmar',
                          storage='sqlite:///params.db')
params = study.best_trial.params

print("Training PPO2 agent with params:", params)
print("Best trial:", study.best_trial.value)

df = pd.read_csv('./data/coinbase_hourly.csv')
df = df.drop(['Symbol'], axis=1)
df = df.sort_values(['Date'])
df = add_indicators(df.reset_index())

test_len = int(len(df) * 0.2)
train_len = int(len(df)) - test_len

train_df = df[:train_len]
test_df = df[train_len:]

train_env = DummyVecEnv([lambda: BitcoinTradingEnv(
    train_df, reward_func="calmar", forecast_len=int(params['forecast_len']), confidence_interval=params['confidence_interval'])])

test_env = DummyVecEnv([lambda: BitcoinTradingEnv(
    test_df, reward_func="calmar", forecast_len=int(params['forecast_len']), confidence_interval=params['confidence_interval'])])

model_params = {
    'n_steps': int(params['n_steps']),
    'gamma': params['gamma'],
    'learning_rate': params['learning_rate'],
    'ent_coef': params['ent_coef'],
    'cliprange': params['cliprange'],
    'noptepochs': int(params['noptepochs']),
    'lam': params['lam'],
}

curr_idx = -1
model = PPO2(MlpLnLstmPolicy, train_env, verbose=0, nminibatches=1,
            tensorboard_log="./tensorboard", **model_params)

# curr_idx = 2
# model = PPO2.load('./agents/ppo2_calmar_' + str(curr_idx) + '.pkl', env=train_env)

for idx in range(curr_idx + 1, 5):
    print('[', idx, '] Training for: ', train_len, ' time steps')

    model.learn(total_timesteps=train_len)

    obs = test_env.reset()
    done, reward_sum = False, 0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        reward_sum += reward

    print('[', idx, '] Total reward: ', reward_sum, ' (calmar)')
    model.save('./agents/ppo2_calmar_' + str(idx) + '.pkl')
