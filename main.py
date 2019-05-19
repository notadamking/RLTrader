import gym
import optuna
import pandas as pd

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2

from env.BitcoinTradingEnv import BitcoinTradingEnv

study = optuna.load_study(study_name='optimize_profit',
                          storage='sqlite:///agents.db')
params = study.best_trial.params

print(params)

df = pd.read_csv('./data/coinbase_hourly.csv')
df = df.drop(['Symbol'], axis=1)
df = df.sort_values(['Date'])

test_len = int(len(df) * 0.2)
train_len = 100  # int(len(df)) - test_len

train_df = df[:train_len]
test_df = df[train_len:]

train_env = DummyVecEnv([lambda: BitcoinTradingEnv(
    train_df, n_forecasts=int(params['n_forecasts']), confidence_interval=params['confidence_interval'])])

model_params = {
    'n_steps': int(params['n_steps']),
    'gamma': params['gamma'],
    'learning_rate': params['learning_rate'],
    'ent_coef': params['ent_coef'],
    'cliprange': params['cliprange'],
    'noptepochs': int(params['noptepochs']),
    'lam': params['lam']
}

model = PPO2(MlpLstmPolicy, train_env, verbose=1, nminibatches=1,
             tensorboard_log="./tensorboard", **model_params)
model.learn(total_timesteps=train_len)

test_env = DummyVecEnv([lambda: BitcoinTradingEnv(
    test_df, n_forecasts=int(params['n_forecasts']), confidence_interval=params['confidence_interval'])])

obs = test_env.reset()
for i in range(test_len):
    action, _states = model.predict(obs)
    obs, rewards, done, info = test_env.step(action)

    test_env.render(mode="human")

test_env.close()
