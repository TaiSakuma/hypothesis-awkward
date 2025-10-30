from functools import partial
from typing import Any, TypeAlias

import numpy as np
from hypothesis import strategies as st

import awkward as ak

from .dtype import builtin_safe_dtypes, items_from_dtype

NestedList: TypeAlias = 'list[Any | NestedList]'


@st.composite
def lists(
    draw: st.DrawFn,
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,
    max_size: int = 5,
    max_depth: int = 5,
) -> NestedList:
    '''Strategy for nested Python lists for which Awkward Arrays can be created.

    Parameters
    ----------
    dtype
        The dtype of the list items or a strategy to generate it. If `None`, a dtype is
        drawn from `builtin_safe_dtypes()`.
    allow_nan
        Generate potentially `NaN` for relevant dtypes if `True`.
    max_size
        Maximum length of any list at any level of nesting.
    max_depth
        Maximum depth of nested lists.

    Examples
    --------
    >>> lists().example()
    [...]
    '''

    if dtype is None:
        dtype = draw(builtin_safe_dtypes())
    if isinstance(dtype, st.SearchStrategy):
        dtype = draw(dtype)
    items = items_from_dtype(dtype, allow_nan=allow_nan)
    extend = partial(st.lists, max_size=max_size)
    max_leaves = max_depth - 1  # `-1` for the outermost `extend`
    if max_leaves <= 0:
        return draw(extend(items))
    return draw(extend(st.recursive(base=items, extend=extend, max_leaves=max_leaves)))


def from_list(
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,
    max_size: int = 5,
    max_depth: int = 5,
) -> st.SearchStrategy[ak.Array]:
    '''Strategy for Awkward Arrays created from Python lists.

    Parameters
    ----------
    dtype
        The dtype of the list items or a strategy to generate it. If `None`, a dtype is
        drawn from `builtin_safe_dtypes()`.
    allow_nan
        Generate potentially `NaN` for relevant dtypes if `True`.
    max_size
        Maximum length of any list at any level of nesting.
    max_depth
        Maximum depth of nested lists.

    Examples
    --------
    >>> from_list().example()
    <Array ... type='...'>
    '''
    return st.builds(
        ak.Array,
        lists(dtype=dtype, allow_nan=allow_nan, max_size=max_size, max_depth=max_depth),
    )
