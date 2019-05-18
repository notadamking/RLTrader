import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()

df = pd.read_csv('./data/coinbase_daily.csv')
df = df.dropna().reset_index().sort_values('Date')

scaled_df = df[['Open', 'Close', 'High', 'Low']].values
scaled_df = scaler.fit_transform(scaled_df.astype('float64'))
scaled_df = pd.DataFrame(scaled_df, columns=['Open', 'Close', 'High', 'Low'])

df['normalized'] = scaled_df['Close']
df['diffed'] = df['Close'] - df['Close'].shift(1)
df['logged'] = np.log(df['Close'])
df['logged_and_diffed'] = df['logged'] - df['logged'].shift(1)

normalized_result = adfuller(df['normalized'].values[1:], autolag="AIC")
print('ADF Statistic: %f' % normalized_result[0])
print('p-value: %f' % normalized_result[1])
print('Critical Values:')
for key, value in normalized_result[4].items():
    print('\t%s: %.3f' % (key, value))

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

plt.title("Normalized Price")
plt.plot(df['Date'], df['normalized'])

plt.show()
