
from lib.env import TradingEnv
from lib.env.exchange import BaseExchange

class LiveExchange(BaseExchange):

    def __init__(self, env: TradingEnv, **kwargs):
        self.env = env
        self.credentials = kwargs.get('credentials')

    def get_account_history(self):
        pass

    def buy(self, amount: float):
        pass

    def sell(self, amount: float):
        pass
