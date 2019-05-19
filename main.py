import gym
import pandas as pd

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2

from env.BitcoinTradingEnv import BitcoinTradingEnv

df = pd.read_csv('./data/coinbase_hourly.csv')
df = df.drop(['Symbol'], axis=1)

test_len = int(len(df) * 0.2)
train_len = int(len(df)) - test_len

train_df = df[:train_len]
test_df = df[train_len:]

train_env = DummyVecEnv([lambda: BitcoinTradingEnv(train_df)])

model = PPO2(MlpLstmPolicy, train_env, verbose=1, nminibatches=1,
             tensorboard_log="./tensorboard")
model.learn(total_timesteps=train_len)

test_env = DummyVecEnv([lambda: BitcoinTradingEnv(test_df)])

obs = test_env.reset()
for i in range(test_len):
    action, _states = model.predict(obs)
    obs, rewards, done, info = test_env.step(action)

    test_env.render(mode="system", title="BTC")

test_env.close()
