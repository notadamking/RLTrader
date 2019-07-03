
from lib.env.TradingEnv import TradingEnv
from lib.data.providers.dates import ProviderDateFormat
from lib.data.providers import StaticDataProvider


# The data with obs and normalized

input_data_path = '../../data/input/coinbase_hourly.csv'

data_columns = {'Date': 'Date', 'Open': 'Open', 'High': 'High',
                'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume BTC'}

data_provider = StaticDataProvider(date_format=ProviderDateFormat.DATETIME_HOUR_12,
                                        csv_data_path=input_data_path,
                                        data_columns=data_columns)

env = TradingEnv(data_provider)

observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

    print("###############################")
    print(action)
    print(str(observation) + str(reward) + str(done) + str(info))
    print("###############################")

    if done:
        observation = env.reset()
env.close()