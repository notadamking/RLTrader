
import abc
import pandas as pd

from enum import Enum
from lib.env import TradingEnv


class BaseExchange(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, env: TradingEnv, **kwargs):
        pass

    @abc.abstractmethod
    def get_account_history(self):
        raise NotImplementedError

    @abc.abstractmethod
    def buy(self, amount: float):
        raise NotImplementedError

    @abc.abstractmethod
    def sell(self, amount: float):
        raise NotImplementedError

    @abc.abstractmethod
    def hold(self):
        raise NotImplementedError
