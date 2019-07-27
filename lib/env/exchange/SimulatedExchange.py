
from lib.env import TradingEnv


class SimulatedExchange(BaseExchange):

    def __init__(self, env: TradingEnv, initial_balance: int = 10000, **kwargs):
        self.env = env
        self.initial_balance = round(initial_balance, self.base_precision)
        self.balance = self.initial_balance
        self.net_worths = [self.initial_balance]
        self.asset_held = 0
        self.current_step = 0
        self.account_history = pd.DataFrame([{
            'balance': self.balance,
            'asset_bought': 0,
            'cost_of_asset': 0,
            'asset_sold': 0,
            'revenue_from_sold': 0,
        }])
        self.trades = []

    def get_account_history(self):
        return self.account_history

    def _add_trade(self, amount: float, total: float, side: str):
        self.trades.append({'step': self.env.current_step,
                            'amount': asset_bought,
                            'total': purchase_cost,
                            'type': side})
        current_net_worth = self.balance + self.asset_held * self.env._current_price()
        current_net_worth = round(current_net_worth, self.base_precision)
        self.net_worths.append(current_net_worth)

    def buy(self, amount: float):
        trade = self.env.trade_strategy.trade(buy_amount=amount,
                                              sell_amount=0,
                                              balance=self.balance,
                                              asset_held=self.asset_held,
                                              current_price=self.env._current_price)
        asset_bought, asset_sold, purchase_cost, sale_revenue = trade
        self.asset_held += asset_bought
        self.balance -= purchase_cost
        self._add_trade(amount=asset_bought, total=purchase_cost, side='buy')
        self.account_history = self.account_history.append({
            'balance': self.balance,
            'asset_held': self.asset_held,
            'asset_bought': asset_bought,
            'purchase_cost': purchase_cost,
            'asset_sold': asset_sold,
            'sale_revenue': sale_revenue,
        }, ignore_index=True)

        self.current_step += 1
        self.current_ohlcv = self.data_provider.next_ohlcv()

    def sell(self, amount: float):
        trade = self.env.trade_strategy.trade(buy_amount=0,
                                              sell_amount=amount,
                                              balance=self.balance,
                                              asset_held=self.asset_held,
                                              current_price=self.env._current_price)
        asset_bought, asset_sold, purchase_cost, sale_revenue = trade
        self.asset_held -= asset_sold
        self.balance += sale_revenue
        self._add_trade(amount=asset_sold, total=sale_revenue, side='sell')
        self.account_history = self.account_history.append({
            'balance': self.balance,
            'asset_bought': asset_bought,
            'cost_of_asset': cost_of_asset,
            'asset_sold': asset_sold,
            'revenue_from_sold': revenue_from_sold,
        }, ignore_index=True)

    def reset(self):
        self.balance = self.initial_balance
        self.net_worths = [self.initial_balance]
        self.asset_held = 0
        self.current_step = 0
        self.last_bought = 0
        self.last_sold = 0
        self.account_history = pd.DataFrame([{
            'balance': self.balance,
            'asset_bought': 0,
            'cost_of_asset': 0,
            'asset_sold': 0,
            'revenue_from_sold': 0,
        }])
        self.trades = []
