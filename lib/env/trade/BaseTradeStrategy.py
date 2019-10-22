from abc import ABCMeta, abstractmethod
from typing import Tuple, Callable


class BaseTradeStrategy(object, metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self,
                 commissionPercent: float,
                 maxSlippagePercent: float,
                 base_precision: int,
                 asset_precision: int,
                 min_cost_limit: float,
                 min_amount_limit: float):
        pass

    @abstractmethod
    def trade(self,
              buy_amount: float,
              sell_amount: float,
              balance: float,
              asset_held: float,
              current_price: Callable[[str], float]) -> Tuple[float, float, float, float]:
        raise NotImplementedError()
