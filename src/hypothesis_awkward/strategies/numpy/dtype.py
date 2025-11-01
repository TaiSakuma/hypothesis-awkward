import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

from hypothesis_awkward.util import (
    SUPPORTED_DTYPE_NAMES,
    SUPPORTED_DTYPES,
    n_scalars_in,
)


def supported_dtype_names() -> st.SearchStrategy[str]:
    '''Strategy for names of NumPy dtypes supported by Awkward Array.

    Examples
    --------
    >>> supported_dtype_names().example()
    '...'
    '''
    return st.sampled_from(SUPPORTED_DTYPE_NAMES)


def supported_dtypes() -> st.SearchStrategy[np.dtype]:
    '''Strategy for NumPy dtypes supported by Awkward Array.

    Examples
    --------
    >>> supported_dtypes().example()
    dtype(...)
    '''
    return st.sampled_from(SUPPORTED_DTYPES)


def numpy_dtypes(
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None = None,
    allow_array: bool = True,
    max_size: int = 5,
) -> st.SearchStrategy[np.dtype]:
    '''Strategy for dtypes (simple or array) supported by Awkward Array.

    Examples of simple dtypes are dtype('int32'), dtype('float64')

    Array dtypes include ([('f0', 'i4'), ('f1', 'f8')]). They are dtypes of structured
    NumPy arrays.

    Parameters
    ----------
    dtype
        A simple dtype or a strategy for simple dtypes for determining the type of
        array elements. If `None`, any supported simple dtype is used.
    allow_array
        Generate only simple dtypes if `False`, else array dtypes as well.
    max_size
        Maximum number of scalars in a structured dtype.

    Examples
    --------
    >>> numpy_dtypes().example()
    dtype(...)
    '''
    if dtype is None:
        dtype = supported_dtypes()
    if not isinstance(dtype, st.SearchStrategy):
        dtype = st.just(dtype)
    if not allow_array:
        return dtype
    return st_np.array_dtypes(
        subtype_strategy=dtype, max_size=max_size, allow_subarrays=True
    ).filter(lambda d: n_scalars_in(d) <= max_size)
