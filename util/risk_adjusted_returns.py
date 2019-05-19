'''

A large part of the code in this file was sourced from the turningfinance blog post.
It has since been heavily modified, but you can consult the post for more information.

Source: http://www.turingfinance.com/computational-investing-with-python-week-one/

'''

import math
import numpy as np

'''
Note - for some of the metrics the absolute value is returns. This is because if the risk (loss) is higher we want to
discount the expected excess return from the portfolio by a higher amount. Therefore risk should be positive.
'''


def beta(returns, market):
    market_returns = np.matrix([returns, market])
    return np.cov(market_returns)[0][1] / max(np.std(market), 1E-10)


def treynor_ratio(returns, benchmark, market):
    return sum(returns - benchmark) / max(beta(returns, market), 1E-10)


def sharpe_ratio(returns, benchmark):
    return sum(returns - benchmark) / max(np.std(returns), 1E-10)


def information_ratio(returns, benchmark):
    diff = returns - benchmark
    return np.mean(diff) / max(np.std(diff), 1E-10)


def modigliani_ratio(returns, benchmark):
    return sharpe_ratio(returns, benchmark) * np.std(benchmark)


def partial_moment(returns, threshold, order, direction):
    threshold_array = np.full(len(returns), threshold)

    if direction == 'lower':
        diff = threshold_array - returns
    elif direction == 'higher':
        diff = returns - threshold_array

    diff = diff.clip(min=0)

    return np.average(diff ** order)


def lower_partial_moment(returns, threshold, order):
    return partial_moment(returns, threshold, order, 'lower')


def higher_partial_moment(returns, threshold, order):
    return partial_moment(returns, threshold, order, 'higher')


def sortino_ratio(returns, benchmark, target=0):
    return sum(returns - benchmark) / max(math.sqrt(lower_partial_moment(returns, target, 2)), 1E-10)


def kappa_three_ratio(returns, benchmark, target=0):
    return sum(returns - benchmark) / max(math.pow(lower_partial_moment(returns, target, 3), float(1/3)), 1E-10)


def omega_ratio(returns, target=0):
    return higher_partial_moment(returns, target, 1) / max(lower_partial_moment(returns, target, 1), 1E-10)


def upside_potential_ratio(returns, target=0):
    return higher_partial_moment(returns, target, 1) / max(math.sqrt(lower_partial_moment(returns, target, 2)), 1E-10)


def drawdowns(returns):
    sums = returns.cumsum()
    top = sums.cummax()
    return sums - top


def calmar_ratio(returns, benchmark):
    return (returns - benchmark) / max(max(drawdowns(returns)), 1E-10)


def sterling_ratio(returns, benchmark):
    return (returns - benchmark) / max(np.average(drawdowns(returns)), 1E-10)


def burke_ratio(returns, benchmark, periods):
    return (returns - benchmark) / max(math.sqrt(np.average([math.sqrt(ret) for ret in drawdowns(returns)])), 1E-10)


def val_at_risk(returns, alpha):
    index = int(alpha * len(returns))
    return abs(np.sort(returns)[index])


def excess_val_at_risk(returns, benchmark, alpha):
    return (returns - benchmark) / max(val_at_risk(returns, alpha), 1E-10)


def cond_val_at_risk(returns, alpha):
    index = int(alpha * len(returns))
    return abs(sum(np.sort(returns)) / index)


def conditional_sharpe_ratio(returns, benchmark, alpha):
    return (returns - benchmark) / max(cond_val_at_risk(returns, alpha), 1E-10)
