from collections import deque

import pandas as pd
import numpy as np
from typing import List, Callable

from lib.env.reward.BaseRewardStrategy import BaseRewardStrategy


class WeightedUnrealizedProfit(BaseRewardStrategy):
    def __init__(self, **kwargs):
        self.decay_rate = kwargs.get('decay_rate', 1e-2)
        self.decay_denominator = np.exp(-1 * self.decay_rate)

        self.reset_reward()

    def reset_reward(self):
        self.rewards = deque(np.zeros(1, dtype=float))
        self.sum = 0.0

    def calc_reward(self, reward):
        self.sum = self.sum - self.decay_denominator * self.rewards.popleft()
        self.sum = self.sum * self.decay_denominator
        self.sum = self.sum + reward

        self.rewards.append(reward)

        return self.sum / self.decay_denominator

    def get_reward(self,
                   current_step: int,
                   current_price: Callable[[str], float],
                   observations: pd.DataFrame,
                   account_history: pd.DataFrame,
                   net_worths: List[float]) -> float:
        if account_history['asset_sold'].values[-1] > 0:
            reward = self.calc_reward(account_history['sale_revenue'].values[-1])
        else:
            reward = self.calc_reward(account_history['asset_held'].values[-1] * current_price())

        return reward
