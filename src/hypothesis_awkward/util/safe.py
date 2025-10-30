from collections.abc import Iterable
from typing import Any, Optional, TypeVar

T = TypeVar('T')


def safe_min(vals: Iterable[T], default: Optional[T] = None) -> Optional[T]:
    '''The smallest item in `vals` that is not `None`.

    >>> safe_min([None, 1, 2, None])
    1

    It returns `None` if `vals` is empty or all items in `vals` are `None`.
    >>> print(safe_min([None, None]))
    None

    >>> print(safe_min([]))
    None

    If `default` is given, it returns `default` instead of `None`.
    >>> safe_min([None, None], default=-1)
    -1

    >>> safe_min([], default=-1)
    -1
    '''
    return min((v for v in vals if v is not None), default=default)  # type: ignore


def safe_max(vals: Iterable[T], default: Optional[T] = None) -> Optional[T]:
    '''The largest item in `vals` that is not `None`.

    >>> safe_max([None, 1, 2, None])
    2

    It returns `None` if `vals` is empty or all items in `vals` are `None`.
    >>> print(safe_max([None, None]))
    None

    >>> print(safe_max([]))
    None

    If `default` is given, it returns `default` instead of `None`.
    >>> safe_max([None, None], default=-1)
    -1

    >>> safe_max([], default=-1)
    -1
    '''
    return max((v for v in vals if v is not None), default=default)  # type: ignore


class GreaterAndLessThanAny:
    '''True for all inequality comparisons.

    >>> GreaterAndLessThanAny() < 1
    True

    >>> GreaterAndLessThanAny() > 1
    True

    >>> GreaterAndLessThanAny() <= 1
    True

    >>> GreaterAndLessThanAny() >= 1
    True

    '''

    def __le__(self, _: Any) -> bool:
        return True

    def __lt__(self, _: Any) -> bool:
        return True

    def __ge__(self, _: Any) -> bool:
        return True

    def __gt__(self, _: Any) -> bool:
        return True

    # def __eq__(self, _: Any) -> bool:
    #     return True

    def __repr__(self) -> str:
        return 'GreaterAndLessThanAny()'


def safe_compare(value: T | None) -> T | GreaterAndLessThanAny:
    '''The `value` itself if it is not `None`. Otherwise, an object that returns True for all inequality comparisons.

    This function helps you concisely write assertions that compare values that
    may be `None`.

    Suppose you have `min_` and `max_` that may be `None`
    >>> import random
    >>> min_ = random.choice([None, 1])
    >>> max_ = random.choice([None, 3])

    and `val` that should be in the range `[min_, max_]`:
    >>> val = 2

    Without this function, you need to check if `min_` and `max_` are `None`.

    >>> if min_ is not None:
    ...   assert min_ <= val

    >>> if max_ is not None:
    ...    assert val <= max_

    This function lets you write the same assertion in one line:

    >>> assert safe_compare(min_) <= val <= safe_compare(max_)

    '''
    if value is None:
        return GreaterAndLessThanAny()
    return value
