import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.BitcoinTradingEnv import BitcoinTradingEnv

import pandas as pd

df = pd.read_csv('./data/bitstamp.csv')
df = df.sort_values('Timestamp')

slice_point = int(len(df) - 50000)

train_df = df[:slice_point]
test_df = df[slice_point:]

train_env = DummyVecEnv(
    [lambda: BitcoinTradingEnv(train_df, serial=True)])

model = PPO2(MlpPolicy, train_env, verbose=1, tensorboard_log="./tensorboard/")
model.learn(total_timesteps=200000)

test_env = DummyVecEnv(
    [lambda: BitcoinTradingEnv(test_df, serial=True)])

obs = test_env.reset()
for i in range(50000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = test_env.step(action)
    test_env.render(mode="system", title="BTC")

test_env.close()
