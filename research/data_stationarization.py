import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv('./data/bitstamp.csv')
df = df.dropna().reset_index().sort_values('Timestamp')
df = df[-1000000:]

df['diffed'] = df['Close'] - df['Close'].shift(1)
df['logged'] = np.log(df['Close'])
df['logged_and_diffed'] = df['logged'] - df['logged'].shift(1)

diffed_result = adfuller(df['diffed'].values[1:], autolag="AIC")
print('ADF Statistic: %f' % diffed_result[0])
print('p-value: %f' % diffed_result[1])
print('Critical Values:')
for key, value in diffed_result[4].items():
    print('\t%s: %.3f' % (key, value))

logged_result = adfuller(df['logged'].values[1:], autolag="AIC")
print('ADF Statistic: %f' % logged_result[0])
print('p-value: %f' % logged_result[1])
print('Critical Values:')
for key, value in logged_result[4].items():
    print('\t%s: %.3f' % (key, value))

logged_and_diffed_result = adfuller(
    df['logged_and_diffed'].values[1:], autolag="AIC")
print('ADF Statistic: %f' % logged_and_diffed_result[0])
print('p-value: %f' % logged_and_diffed_result[1])
print('Critical Values:')
for key, value in logged_and_diffed_result[4].items():
    print('\t%s: %.3f' % (key, value))

plt.title("Close Price")
plt.plot(df['Timestamp'], df['Close'])

plt.show()
