import pandas as pd

from typing import List

from lib.env.reward.BaseRewardStrategy import BaseRewardStrategy


class IncrementalProfit(BaseRewardStrategy):
    def __init__(self):
        super(IncrementalProfit, self).__init__()

    def reset_reward(self):
        pass

    def get_reward(self,
                   observations: pd.DataFrame,
                   account_history: pd.DataFrame,
                   net_worths: List[float],
                   last_bought: int,
                   last_held: int,
                   last_sold: int,
                   current_price: float):
        curr_balance = account_history['balance'].values[-1]
        prev_balance = account_history['balance'].values[-2] if len(account_history['balance']) > 1 else curr_balance
        reward = 0

        if curr_balance > prev_balance:
            reward = net_worths[-1] - net_worths[last_bought]
        elif curr_balance < prev_balance:
            reward = observations['Close'].values[last_sold] - current_price

        return reward
