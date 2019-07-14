from typing import Tuple, Callable
from enum import Enum

from lib.env.trade import BaseTradeStrategy


class LiveTradeStrategy(BaseTradeStrategy):
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
        raise NotImplementedError()
