import numpy as np

from typing import Tuple, Callable

from lib.env.trade import BaseTradeStrategy


class SimulatedTradeStrategy(BaseTradeStrategy):
    def __init__(self,
                 commissionPercent: float,
                 maxSlippagePercent: float,
                 base_precision: int,
                 asset_precision: int,
                 min_cost_limit: float,
                 min_amount_limit: float):
        self.commissionPercent = commissionPercent
        self.maxSlippagePercent = maxSlippagePercent
        self.base_precision = base_precision
        self.asset_precision = asset_precision
        self.min_cost_limit = min_cost_limit
        self.min_amount_limit = min_amount_limit

    def trade(self,
              buy_amount: float,
              sell_amount: float,
              balance: float,
              asset_held: float,
              current_price: Callable[[str], float]) -> Tuple[float, float, float, float]:
        current_price = current_price('Close')
        commission = self.commissionPercent / 100
        slippage = np.random.uniform(0, self.maxSlippagePercent) / 100

        asset_bought, asset_sold, purchase_cost, sale_revenue = buy_amount, sell_amount, 0, 0

        if buy_amount > 0 and balance >= self.min_cost_limit:
            price_adjustment = (1 + commission) * (1 + slippage)
            buy_price = round(current_price * price_adjustment, self.base_precision)
            purchase_cost = round(buy_price * buy_amount, self.base_precision)
        elif sell_amount > 0 and asset_held >= self.min_amount_limit:
            price_adjustment = (1 - commission) * (1 - slippage)
            sell_price = round(current_price * price_adjustment, self.base_precision)
            sale_revenue = round(sell_amount * sell_price, self.base_precision)

        return asset_bought, asset_sold, purchase_cost, sale_revenue
