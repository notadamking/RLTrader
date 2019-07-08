import pandas as pd

from abc import ABCMeta, abstractmethod
from typing import List


class BaseRewardStrategy(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_reward(self,
                   observations: pd.DataFrame,
                   account_history: pd.DataFrame,
                   net_worths: List[float],
                   last_bought: int,
                   last_sold: int,
                   current_price: float) -> float:
        raise NotImplementedError()
