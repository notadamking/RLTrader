import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ta

df = pd.read_csv('./data/coinbase_daily.csv')
df = df.dropna().reset_index().sort_values('Date')

ta_df = pd.DataFrame()

ta_df['RSI'] = ta.rsi(df["Close"])
ta_df['MFI'] = ta.money_flow_index(
    df["High"], df["Low"], df["Close"], df["Volume BTC"])
ta_df['TSI'] = ta.tsi(df["Close"])
ta_df['UO'] = ta.uo(df["High"], df["Low"], df["Close"])
ta_df['Stoch'] = ta.stoch(df["High"], df["Low"], df["Close"])
ta_df['Stoch_Signal'] = ta.stoch_signal(df["High"], df["Low"], df["Close"])
ta_df['WR'] = ta.wr(df["High"], df["Low"], df["Close"])
ta_df['AO'] = ta.ao(df["High"], df["Low"])

ta_df['MACD'] = ta.macd(df["Close"])
ta_df['MACD_signal'] = ta.macd_signal(df["Close"])
ta_df['MACD_diff'] = ta.macd_diff(df["Close"])
ta_df['EMA_fast'] = ta.ema_indicator(df["Close"])
ta_df['EMA_slow'] = ta.ema_indicator(df["Close"])
ta_df['Vortex_pos'] = ta.vortex_indicator_pos(
    df["High"], df["Low"], df["Close"])
ta_df['Vortex_neg'] = ta.vortex_indicator_neg(
    df["High"], df["Low"], df["Close"])
ta_df['Vortex_diff'] = abs(
    ta_df['Vortex_pos'] -
    ta_df['Vortex_neg'])
ta_df['Trix'] = ta.trix(df["Close"])
ta_df['Mass_index'] = ta.mass_index(df["High"], df["Low"])
ta_df['CCI'] = ta.cci(df["High"], df["Low"], df["Close"])
ta_df['DPO'] = ta.dpo(df["Close"])
ta_df['KST'] = ta.kst(df["Close"])
ta_df['KST_sig'] = ta.kst_sig(df["Close"])
ta_df['KST_diff'] = (
    ta_df['KST'] -
    ta_df['KST_sig'])
ta_df['Ichimoku_a'] = ta.ichimoku_a(df["High"], df["Low"], visual=True)
ta_df['Ichimoku_b'] = ta.ichimoku_b(df["High"], df["Low"], visual=True)
ta_df['Aroon_up'] = ta.aroon_up(df["Close"])
ta_df['Aroon_down'] = ta.aroon_down(df["Close"])
ta_df['Aroon_ind'] = (
    ta_df['Aroon_up'] -
    ta_df['Aroon_down']
)

ta_df['ATR'] = ta.average_true_range(
    df["High"],
    df["Low"],
    df["Close"])
ta_df['BBH'] = ta.bollinger_hband(df["Close"])
ta_df['BBL'] = ta.bollinger_lband(df["Close"])
ta_df['BBM'] = ta.bollinger_mavg(df["Close"])
ta_df['BBHI'] = ta.bollinger_hband_indicator(
    df["Close"])
ta_df['BBLI'] = ta.bollinger_lband_indicator(
    df["Close"])
ta_df['KCC'] = ta.keltner_channel_central(
    df["High"],
    df["Low"],
    df["Close"])
ta_df['KCH'] = ta.keltner_channel_hband(
    df["High"],
    df["Low"],
    df["Close"])
ta_df['KCL'] = ta.keltner_channel_lband(
    df["High"],
    df["Low"],
    df["Close"])
ta_df['KCHI'] = ta.keltner_channel_hband_indicator(df["High"],
                                                   df["Low"],
                                                   df["Close"])
ta_df['KCLI'] = ta.keltner_channel_lband_indicator(df["High"],
                                                   df["Low"],
                                                   df["Close"])
ta_df['DCH'] = ta.donchian_channel_hband(
    df["Close"])
ta_df['DCL'] = ta.donchian_channel_lband(
    df["Close"])
ta_df['DCHI'] = ta.donchian_channel_hband_indicator(df["Close"])
ta_df['DCLI'] = ta.donchian_channel_lband_indicator(df["Close"])

ta_df['ADI'] = ta.acc_dist_index(df["High"],
                                 df["Low"],
                                 df["Close"],
                                 df["Volume BTC"])
ta_df['OBV'] = ta.on_balance_volume(df["Close"],
                                    df["Volume BTC"])
ta_df['OBVM'] = ta.on_balance_volume_mean(
    df["Close"],
    df["Volume BTC"])
ta_df['CMF'] = ta.chaikin_money_flow(df["High"],
                                     df["Low"],
                                     df["Close"],
                                     df["Volume BTC"])
ta_df['FI'] = ta.force_index(df["Close"],
                             df["Volume BTC"])
ta_df['EM'] = ta.ease_of_movement(df["High"],
                                  df["Low"],
                                  df["Close"],
                                  df["Volume BTC"])
ta_df['VPT'] = ta.volume_price_trend(df["Close"],
                                     df["Volume BTC"])
ta_df['NVI'] = ta.negative_volume_index(df["Close"],
                                        df["Volume BTC"])

ta_df['DR'] = ta.daily_return(df["Close"])
ta_df['DLR'] = ta.daily_log_return(df["Close"])
ta_df['CR'] = ta.cumulative_return(df["Close"])

corr = ta_df.corr()

corr.describe().to_csv('All_indicators.csv')

sns.heatmap(corr,
            cmap="viridis",
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

plt.title("All Indicators")
plt.show()

'''

EMA fast = EMA slow = Ichimoku a = Ichimoku b= ATR = BBH = BBL = BBM = KCC = KCH = KCL = DCH = DCL = CR

RSI ~= Stoch = Stoch_Signal = WR

AO = MACD = MACD_Signal

'''
