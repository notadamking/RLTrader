import pandas as pd

from typing import List

from lib.env.reward import BaseRewardStrategy


class IncrementalProfit(BaseRewardStrategy):
    @staticmethod
    def get_reward(observations: pd.DataFrame,
                   account_history: pd.DataFrame,
                   net_worths: List[float],
                   last_bought: int,
                   last_sold: int,
                   current_price: float):
        prev_balance = account_history['balance'].values[-2]
        curr_balance = account_history['balance'].values[-1]
        reward = 0

        if curr_balance > prev_balance:
            reward = net_worths[-1] - net_worths[last_bought]
        elif curr_balance < prev_balance:
            reward = observations['Close'].values[last_sold] - current_price

        return reward
