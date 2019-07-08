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
