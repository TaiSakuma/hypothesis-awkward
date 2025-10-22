from functools import partial
from typing import TypeVar

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

import awkward as ak

T = TypeVar('T', bound=np.generic)

# NOTE: `datetime64[us]` isn't entirely safe. For example, a value with the year zero is
# coerced to `int`: `np.datetime64('0000-12-31').item() = -719163`.
BUILTIN_SAFE_DTYPE_NAMES = (
    'bool',
    'int64',
    'float64',
    'complex128',
    'datetime64[us]',
    'timedelta64[us]',
)

BUILTIN_SAFE_DTYPES = tuple(np.dtype(name) for name in BUILTIN_SAFE_DTYPE_NAMES)


def builtin_safe_dtype_names() -> st.SearchStrategy[str]:
    '''Strategy for names of NumPy dtypes with corresponding Python built-in types.

    Examples
    --------
    >>> builtin_safe_dtype_names().example()
    '...'
    '''
    return st.sampled_from(BUILTIN_SAFE_DTYPE_NAMES)


def builtin_safe_dtypes() -> st.SearchStrategy[np.dtype]:
    '''Strategy for NumPy dtypes with corresponding Python built-in types.

    Examples
    --------
    >>> builtin_safe_dtypes().example()
    dtype(...)
    '''
    return st.sampled_from(BUILTIN_SAFE_DTYPES)


def items_from_dtype(
    dtype: np.dtype[T], allow_nan: bool = False
) -> st.SearchStrategy[T]:
    '''Strategy for Python built-in type values for a given NumPy dtype.

    Parameters
    ----------
    dtype
        The NumPy dtype to generate items for.
    allow_nan
        Generate potentially `NaN` for relevant dtypes if `True`.

    Examples
    --------
    >>> i = items_from_dtype(np.dtype('int64')).example()
    >>> isinstance(i, int)
    True
    '''
    return (
        st_np.from_dtype(dtype, allow_nan=allow_nan)
        .map(lambda x: x.item())
        .filter(lambda item: dtype.kind == 'i' or type(item) is not int)
    )
    # Reject if the item is coerced to `int` when `dtype` is not integer.
    # This could happen for `datetime64` and `timedelta64` dtypes.


@st.composite
def lists(
    draw: st.DrawFn, allow_nan: bool = False, max_size: int = 5, max_depth: int = 5
) -> list:
    '''Strategy for nested Python lists for which Awkward Arrays can be created.

    Parameters
    ----------
    allow_nan
        Generate potentially `NaN` for relevant dtypes if `True`.
    max_size
        Maximum list length at any depth.
    max_depth
        Maximum depth of nested lists.

    Examples
    --------
    >>> lists().example()
    [...]
    '''

    dtype = draw(builtin_safe_dtypes())
    items = items_from_dtype(dtype, allow_nan=allow_nan)
    extend = partial(st.lists, max_size=max_size)
    max_leaves = max_depth - 1  # `-1` for the outermost `extend`
    if max_leaves <= 0:
        return draw(extend(items))
    return draw(extend(st.recursive(base=items, extend=extend, max_leaves=max_leaves)))


def from_list(
    allow_nan: bool = False, max_size: int = 5, max_depth: int = 5
) -> st.SearchStrategy[ak.Array]:
    '''Strategy for Awkward Arrays created from Python lists.

    Parameters
    ----------
    allow_nan
        Generate potentially `NaN` for relevant dtypes if `True`.
    max_size
        Maximum list length at any depth.
    max_depth
        Maximum depth of nested lists.

    Examples
    --------
    >>> from_list().example()
    <Array ... type='...'>
    '''
    return st.builds(
        ak.Array,
        lists(allow_nan=allow_nan, max_size=max_size, max_depth=max_depth),
    )
