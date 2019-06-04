import pickle
import pandas as pd

from util.benchmarks import buy_and_hodl, rsi_divergence, sma_crossover

df = pd.read_csv('./data/coinbase_hourly.csv')
df = df.drop(['Symbol'], axis=1)
df = df.sort_values(['Date'])

test_len = int(len(df) * 0.2)
train_len = int(len(df)) - test_len

test_df = df[train_len:]
test_df = test_df.fillna(method='bfill').reset_index()

initial_balance = 10000
commission = 0.0025

buy_and_hodl_net_worths = buy_and_hodl(test_df['Close'], initial_balance, commission)
rsi_divergence_net_worths = rsi_divergence(test_df['Close'], initial_balance, commission)
sma_crossover_net_worths = sma_crossover(test_df['Close'], initial_balance, commission)

with open('./research/results/buy_and_hodl_net_worths.pkl', 'wb') as handle:
    pickle.dump(buy_and_hodl_net_worths, handle)

with open('./research/results/rsi_divergence_net_worths.pkl', 'wb') as handle:
    pickle.dump(rsi_divergence_net_worths, handle)

with open('./research/results/sma_crossover_net_worths.pkl', 'wb') as handle:
    pickle.dump(sma_crossover_net_worths, handle)

    
