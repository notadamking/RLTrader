from collections import deque

import pandas as pd
import numpy as np
from typing import List

from lib.env.reward.BaseRewardStrategy import BaseRewardStrategy


class WeightedUnrealisedProfit(BaseRewardStrategy):
    def __init__(self, **kwargs):
        super(WeightedUnrealisedProfit, self).__init__()

        self.decay_rate = kwargs.get('decay_rate', 1e-2)
        self.rewards = deque(np.zeros(1, dtype=float))
        self.sum = 0.0
        self.denominator = np.exp(-1 * self.decay_rate)

    def reset_reward(self):
        self.rewards = deque(np.zeros(1, dtype=float))
        self.sum = 0.0

    def cal_reward(self, reward):
        stale_reward = self.rewards.popleft()
        self.sum = self.sum - np.exp(-1 * self.decay_rate) * stale_reward
        self.sum = self.sum * np.exp(-1 * self.decay_rate)
        self.sum = self.sum + reward
        self.rewards.append(reward)
        return self.sum / self.denominator

    def get_reward(self,
                   observations: pd.DataFrame,
                   account_history: pd.DataFrame,
                   net_worths: List[float],
                   last_bought: int,
                   last_held: int,
                   last_sold: int,
                   current_price: float):

        if account_history['btc_sold'].values[-1] > 0:
            reward = self.cal_reward(account_history['revenue_from_sold'].values[-1])
        else:
            reward = self.cal_reward(last_held * current_price)

        return reward
