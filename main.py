from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
import matplotlib.pyplot as plt

from env.BitcoinTradingEnv import BitcoinTradingEnv
from src.utils.DataLoader import DataLoader
from src.utils.enum import TECHIND


# Example how to load BTC data online from AlphaVantage
# get free API key from https://www.alphavantage.co/support/#api-key
ALPHA_KEY =""
load = DataLoader(ALPHA_KEY)
#df = load.get_crypto("BTC")
# Same for equity
#df = load.get_stock("MSFT")
# same for technical indicators
#ti = load.get_tech_indicator(stock="MSFT",indicator=TECHIND.TECHIND.BBANDS)
#print(ti.head())


#load bitcoin data from local file
df = load.get_local_data('./data/bitstamp.csv')



# split in train & test
slice_point = int(len(df) - 50000)
train_df = df[:slice_point]
test_df = df[slice_point:]

train_env = DummyVecEnv(
    [lambda: BitcoinTradingEnv(train_df, serial=True)])

model = A2C(MlpPolicy, train_env, verbose=1,
            tensorboard_log="./tensorboard/")
model.learn(total_timesteps=200000)

test_env = DummyVecEnv(
    [lambda: BitcoinTradingEnv(test_df, serial=True)])

obs = test_env.reset()
for i in range(50000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = test_env.step(action)
    test_env.render(mode="system", title="BTC")

test_env.close()
