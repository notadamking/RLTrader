import ta
from enum import Enum


class SIGNALS(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


def trade_strategy(prices, initial_balance, commission, signal_fn):
    net_worths = [initial_balance]
    balance = initial_balance
    amount_held = 0

    for i in range(1, len(prices)):
        if amount_held > 0:
            net_worths.append(balance + amount_held * prices[i])
        else:
            net_worths.append(balance)

        signal = signal_fn(i)

        if signal == SIGNALS.SELL and amount_held > 0:
            balance = amount_held * (prices[i] * (1 - commission))
            amount_held = 0
        elif signal == SIGNALS.BUY and amount_held == 0:
            amount_held = balance / (prices[i] * (1 + commission))
            balance = 0

    return net_worths


def buy_and_hodl(prices, initial_balance, commission):
    def signal_fn(i):
        return SIGNALS.BUY

    return trade_strategy(prices, initial_balance, commission, signal_fn)


def rsi_divergence(prices, initial_balance, commission, period=3):
    rsi = ta.rsi(prices)

    def signal_fn(i):
        if i >= period:
            rsiSum = sum(rsi[i - period:i + 1].diff().cumsum().fillna(0))
            priceSum = sum(prices[i - period:i + 1].diff().cumsum().fillna(0))

            if rsiSum < 0 and priceSum >= 0:
                return SIGNALS.SELL
            elif rsiSum > 0 and priceSum <= 0:
                return SIGNALS.BUY

        return SIGNALS.HOLD

    return trade_strategy(prices, initial_balance, commission, signal_fn)


def sma_crossover(prices, initial_balance, commission):
    macd = ta.macd(prices)

    def signal_fn(i):
        if macd[i] > 0 and macd[i - 1] <= 0:
            return SIGNALS.SELL
        elif macd[i] < 0 and macd[i - 1] >= 0:
            return SIGNALS.BUY

        return SIGNALS.HOLD

    return trade_strategy(prices, initial_balance, commission, signal_fn)
