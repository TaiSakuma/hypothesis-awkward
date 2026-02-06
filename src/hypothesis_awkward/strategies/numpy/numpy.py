import math

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

import awkward as ak
from hypothesis_awkward.util import n_scalars_in

from .dtype import numpy_dtypes

# https://github.com/HypothesisWorks/hypothesis/blob/86d8a4d/hypothesis-python/src/hypothesis/extra/_array_helpers.py#L68
NDIM_MAX = 32


@st.composite
def numpy_arrays(
    draw: st.DrawFn,
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None = None,
    allow_structured: bool = True,
    allow_nan: bool = False,
    min_dims: int = 1,
    max_dims: int | None = None,
    max_size: int = 10,
) -> np.ndarray:
    '''Strategy for NumPy arrays from which Awkward Arrays can be created.

    Parameters
    ----------
    dtype
        A simple dtype or a strategy for simple dtypes for determining the type of
        array elements. If `None`, any supported simple dtype is used.
    allow_structured
        Generate only simple arrays if `False`, else structured arrays as well.
    allow_nan
        Generate potentially `NaN` for relevant dtypes if `True`.
    min_dims
        Minimum number of dimensions.
    max_dims
        Maximum number of dimensions. If `None`, auto-derived from `max_size`.
    max_size
        Maximum number of items in the array.

    Examples
    --------
    >>> n = numpy_arrays().example()
    >>> ak.from_numpy(n)
    <Array ... type='...'>

    '''
    dtype = draw(
        numpy_dtypes(
            dtype=dtype, allow_array=allow_structured, max_size=max(1, max_size)
        )
    )
    dtype_size = n_scalars_in(dtype)
    max_size = max_size // dtype_size
    average_size = max_size // 2

    # Empty arrays must be generated separately because st_np.array_shapes() requires
    # min_side >= 1 by default. The probability of generating an empty array is set to
    # P(empty) = 1 / (1 + average_size), matching Hypothesis st.lists() behavior. For
    # max_size=10: average_size=5, P(empty) = 1/6 ≈ 16.7%
    empty = max_size <= 0 or draw(st.integers(min_value=0, max_value=average_size)) == 0

    shape: tuple[int, ...]
    if empty:
        shape = (0,) + (1,) * (min_dims - 1)
    else:
        max_side_upper = math.floor(max_size ** (1 / min_dims))
        max_side_upper = max(1, max_side_upper)
        max_side = draw(st.integers(min_value=1, max_value=max_side_upper))
        if max_side == 1:
            max_dims_upper = min(NDIM_MAX, max_size)
        else:
            max_dims_upper = min(
                NDIM_MAX, math.floor(math.log(max_size) / math.log(max_side))
            )
        if max_dims is not None:
            max_dims_upper = min(max_dims_upper, max_dims)
        n_dims = draw(
            st.integers(
                min_value=min_dims, max_value=max(min_dims, max_dims_upper)
            )
        )
        shape = draw(
            st_np.array_shapes(
                min_dims=min_dims, max_dims=n_dims, max_side=max_side
            )
        )

    return draw(
        st_np.arrays(dtype=dtype, shape=shape, elements={'allow_nan': allow_nan})
    )


def from_numpy(
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None = None,
    allow_structured: bool = True,
    allow_nan: bool = False,
    regulararray: bool | None = None,
    max_size: int = 10,
) -> st.SearchStrategy[ak.Array]:
    '''Strategy for Awkward Arrays created from NumPy arrays.

    Parameters
    ----------
    dtype
        A simple dtype or a strategy for simple dtypes for determining the type of
        array elements. If `None`, any supported simple dtype is used.
    allow_structured
        Generate only from simple NumPy arrays if `False`, else from structured NumPy
        arrays as well.
    allow_nan
        Generate potentially `NaN` for relevant dtypes if `True`.
    regulararray
        Passed to `ak.from_numpy`. If `None` (default), randomly generates
        `True` or `False`.
    max_size
        Maximum number of items in the array.

    Examples
    --------
    >>> from_numpy().example()
    <Array ... type='...'>

    '''

    reg_array = st.booleans() if regulararray is None else st.just(regulararray)

    return st.builds(
        ak.from_numpy,
        numpy_arrays(
            dtype=dtype,
            allow_structured=allow_structured,
            allow_nan=allow_nan,
            max_size=max_size,
        ),
        regulararray=reg_array,
    )
