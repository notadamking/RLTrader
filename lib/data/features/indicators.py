import ta
import pandas as pd


diff = lambda x, y: x - y
abs_diff = lambda x, y: abs(x - y)


indicators = [
    ('RSI', ta.rsi, ['Close']),
    ('MFI', ta.money_flow_index, ['High', 'Low', 'Close', 'Volume BTC']),
    ('TSI', ta.tsi, ['Close']),
    ('UO', ta.uo, ['High', 'Low', 'Close']),
    ('AO', ta.ao, ['High', 'Close']),
    ('MACDDI', ta.macd_diff, ['Close']),
    ('VIP', ta.vortex_indicator_pos, ['High', 'Low', 'Close']),
    ('VIN', ta.vortex_indicator_neg, ['High', 'Low', 'Close']),
    ('VIDIF', abs_diff, ['VIP', 'VIN']),
    ('TRIX', ta.trix, ['Close']),
    ('MI', ta.mass_index, ['High', 'Low']),
    ('CCI', ta.cci, ['High', 'Low', 'Close']),
    ('DPO', ta.dpo, ['Close']),
    ('KST', ta.kst, ['Close']),
    ('KSTS', ta.kst_sig, ['Close']),
    ('KSTDI', diff, ['KST', 'KSTS']),
    ('ARU', ta.aroon_up, ['Close']),
    ('ARD', ta.aroon_down, ['Close']),
    ('ARI', diff, ['ARU', 'ARD']),
    ('BBH', ta.bollinger_hband, ['Close']),
    ('BBL', ta.bollinger_lband, ['Close']),
    ('BBM', ta.bollinger_mavg, ['Close']),
    ('BBHI', ta.bollinger_hband_indicator, ['Close']),
    ('BBLI', ta.bollinger_lband_indicator, ['Close']),
    ('KCHI', ta.keltner_channel_hband_indicator, ['High', 'Low', 'Close']),
    ('KCLI', ta.keltner_channel_lband_indicator, ['High', 'Low', 'Close']),
    ('DCHI', ta.donchian_channel_hband_indicator, ['Close']),
    ('DCLI', ta.donchian_channel_lband_indicator, ['Close']),
    ('ADI', ta.acc_dist_index, ['High', 'Low', 'Close', 'Volume BTC']),
    ('OBV', ta.on_balance_volume, ['Close', 'Volume BTC']),
    ('CMF', ta.chaikin_money_flow, ['High', 'Low', 'Close', 'Volume BTC']),
    ('FI', ta.force_index, ['Close', 'Volume BTC']),
    ('EM', ta.ease_of_movement, ['High', 'Low', 'Close', 'Volume BTC']),
    ('VPT', ta.volume_price_trend, ['Close', 'Volume BTC']),
    ('NVI', ta.negative_volume_index, ['Close', 'Volume BTC']),
    ('DR', ta.daily_return, ['Close']),
    ('DLR', ta.daily_log_return, ['Close'])
]


def add_indicators(df) -> pd.DataFrame:
    for name, f, arg_names in indicators:
        wrapper = lambda func, args: func(*args)
        args = [df[arg_name] for arg_name in arg_names]
        df[name] = wrapper(f, args)
    df.fillna(method='bfill', inplace=True)
    return df
