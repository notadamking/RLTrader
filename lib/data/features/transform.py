import numpy as np

from abc import abstractmethod
from typing import Callable, Iterable, List


@abstractmethod
def transform(iterable: Iterable, inplace: bool = True, columns: List[str] = None, transform_fn: Callable[[Iterable], Iterable] = None):
    if inplace is True:
        transformed_iterable = iterable
    else:
        transformed_iterable = iterable.copy()

    transformed_iterable = transformed_iterable.fillna(method='bfill')

    if transform_fn is None:
        raise NotImplementedError()

    if columns is None:
        transformed_iterable = transform_fn(transformed_iterable)
    else:
        for column in columns:
            transformed_iterable[column] = transform_fn(
                transformed_iterable[column])

    return transformed_iterable


def max_min_normalize(iterable: Iterable, inplace: bool = True, columns: List[str] = None):
    return transform(iterable, inplace, columns, lambda t_iterable: (t_iterable - t_iterable.min()) / (t_iterable.max() - t_iterable.min()))


def difference(iterable: Iterable, inplace: bool = True, columns: List[str] = None):
    return transform(iterable, inplace, columns, lambda t_iterable: t_iterable - t_iterable.shift(1))


def log_and_difference(iterable: Iterable, inplace: bool = True, columns: List[str] = None):
    return transform(iterable, inplace, columns, lambda t_iterable: np.log(t_iterable) - np.log(t_iterable).shift(1))
