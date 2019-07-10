import pandas as pd

from abc import ABCMeta, abstractmethod
from typing import List, Callable


class BaseRewardStrategy(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reset_reward(self):
        raise NotImplementedError()

    @abstractmethod
    def get_reward(self,
                   current_step: int,
                   current_price: Callable[[str], float],
                   observations: pd.DataFrame,
                   account_history: pd.DataFrame,
                   net_worths: List[float]) -> float:
        raise NotImplementedError()
