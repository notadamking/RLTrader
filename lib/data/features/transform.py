import numpy as np
import pandas as pd

from abc import abstractmethod
from typing import Callable, Iterable, List


@abstractmethod
def transform(iterable: Iterable, inplace: bool = True, columns: List[str] = None, transform_fn: Callable[[Iterable], Iterable] = None):
    if inplace is True:
        transformed_iterable = iterable
    else:
        transformed_iterable = iterable.copy()

    if isinstance(transformed_iterable, pd.DataFrame):
        is_list = False
    else:
        is_list = True
        transformed_iterable = pd.DataFrame(transformed_iterable, columns=columns)

    transformed_iterable.fillna(0, inplace=True)

    if transform_fn is None:
        raise NotImplementedError()

    if columns is None:
        columns = transformed_iterable.columns

    for column in columns:
        transformed_iterable[column] = transform_fn(transformed_iterable[column])

    transformed_iterable.fillna(method="bfill", inplace=True)
    transformed_iterable[np.bitwise_not(np.isfinite(transformed_iterable))] = 0

    if is_list:
        transformed_iterable = transformed_iterable.values

    return transformed_iterable


def max_min_normalize(iterable: Iterable, inplace: bool = True, columns: List[str] = None):
    return transform(iterable, inplace, columns, lambda t_iterable: (t_iterable - t_iterable.min()) / (t_iterable.max() - t_iterable.min()))


def mean_normalize(iterable: Iterable, inplace: bool = True, columns: List[str] = None):
    return transform(iterable, inplace, columns, lambda t_iterable: (t_iterable - t_iterable.mean()) / t_iterable.std())


def difference(iterable: Iterable, inplace: bool = True, columns: List[str] = None):
    return transform(iterable, inplace, columns, lambda t_iterable: t_iterable - t_iterable.shift(1))


def log_and_difference(iterable: Iterable, inplace: bool = True, columns: List[str] = None):
    return transform(iterable, inplace, columns, lambda t_iterable: np.log(t_iterable) - np.log(t_iterable).shift(1))


class FastTransform:

    MIN="min"
    MAX="max"

    def __init__(self):
        self.cached_min_max = dict()

    def max_min_normalize(self, iterable: Iterable, inplace: bool = True, columns: List[str] = None):
        columns = iterable.columns if columns is None else columns

        if iterable.empty:
            return iterable

        for column in columns:
            cached_min = self.cached_min_max[FastTransform.MIN + column] if FastTransform.MIN + column in self.cached_min_max else np.min(iterable[column])
            cached_max = self.cached_min_max[FastTransform.MAX + column] if FastTransform.MAX + column in self.cached_min_max else np.max(iterable[column])

            new_min = min(cached_min, iterable.iloc[-1][column])
            new_max = max(cached_max, iterable.iloc[-1][column])

            iterable.iloc[-1][column] = (iterable.iloc[-1][column] - new_min) / (new_max - new_min)

            self._update_cache(column, new_min, new_max)

        return iterable

    def _update_cache(self,column, min, max):
        #don't want to save nan's as it will force recalculation on the next iteration
        if not np.isnan(min):
            self.cached_min_max[FastTransform.MIN + column] = min

        if not np.isnan(max):
            self.cached_min_max[FastTransform.MAX + column] = max

    def _get_cached(self, column):
        return self.cached_min_max[FastTransform.MIN + column], self.cached_min_max[FastTransform.MAX + column]