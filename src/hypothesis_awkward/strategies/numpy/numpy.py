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
    min_size: int = 0,
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
    min_size
        Minimum number of scalars in the array. For structured dtypes, each
        element counts as multiple scalars (one per field).
    max_size
        Maximum number of scalars in the array. For structured dtypes, each
        element counts as multiple scalars (one per field).

    Examples
    --------
    >>> n = numpy_arrays().example()
    >>> ak.from_numpy(n)
    <Array ... type='...'>

    '''
    # Limit dtype_size so that the size can be between min_size and max_size
    max_dtype_size = max_size // min_size if min_size > 0 else max_size
    max_dtype_size = max(1, max_dtype_size)
    dtype = draw(
        numpy_dtypes(
            dtype=dtype,
            allow_array=allow_structured,
            max_size=max_dtype_size,
        )
    )
    dtype_size = n_scalars_in(dtype)
    min_items = -(-min_size // dtype_size)  # n items of dtype, rounded up
    max_items = max_size // dtype_size  # n items of dtype, rounded down

    shape = draw(_st_shape(min_items, max_items, min_dims, max_dims))

    return draw(
        st_np.arrays(dtype=dtype, shape=shape, elements={'allow_nan': allow_nan})
    )


@st.composite
def _st_shape(
    draw: st.DrawFn,
    min_items: int,
    max_items: int,
    min_dims: int,
    max_dims: int | None,
) -> tuple[int, ...]:
    '''Strategy for a non-empty array shape.

    A shape is a tuple of positive integers, e.g., ``(3, 5, 2)``, where each
    value is the length of a dimension (called a "side").

    Parameters
    ----------
    min_items
        Minimum number of elements, i.e., ``prod(shape) >= min_items``.
    max_items
        Maximum number of elements, i.e., ``prod(shape) <= max_items``.
    min_dims
        Minimum number of dimensions, i.e., ``len(shape) >= min_dims``.
    max_dims
        Maximum number of dimensions. If ``None``, derived from ``max_items``.
    '''

    # Generate empty shape manually as `st_np.array_shapes()` doesn't.
    empty = draw(_st_empty(min_items, max_items))
    if empty:
        return (0,) + (1,) * (min_dims - 1)

    # Bounds of the largest integer in the shape (the "max side")
    root = 1 / min_dims
    min_max_side = max(1, math.ceil(min_items**root)) if min_items > 0 else 1
    max_max_side = max(1, math.floor(max_items**root))

    if not min_max_side <= max_max_side:
        # This can happen typically when `max_max_side` is 1, i.e., `min_dims > 1` and
        # `max_items` is small. We return a shape in which the first dimension holds all
        # items to satisfy the min_items and max_items, and the rest of the dimensions
        # are 1 to satisfy the min_dims and max_dims, e.g., (9, 1, 1).
        size = draw(st.integers(min_value=min_items, max_value=max_items))
        extra_dims = 0
        if max_dims is not None and max_dims > min_dims:
            extra_dims = draw(st.integers(0, max_dims - min_dims))
        return (size,) + (1,) * (min_dims - 1 + extra_dims)

    max_side = draw(st.integers(min_value=min_max_side, max_value=max_max_side))
    if max_side == 1:
        max_max_dims = min(NDIM_MAX, max_items)
    else:
        max_max_dims = min(
            NDIM_MAX, math.floor(math.log(max_items) / math.log(max_side))
        )
    if max_dims is not None:
        max_max_dims = min(max_max_dims, max_dims)
    max_len = draw(
        st.integers(min_value=min_dims, max_value=max(min_dims, max_max_dims))
    )
    min_side = max(1, math.ceil(min_items ** (1 / max_len))) if min_items > 0 else 1
    min_len = max_len if min_items > 0 else min_dims
    return draw(
        st_np.array_shapes(
            min_dims=min_len,
            max_dims=max_len,
            min_side=min_side,
            max_side=max_side,
        )
    )


def _st_empty(min_items: int, max_items: int) -> st.SearchStrategy[bool]:
    '''Strategy for whether to generate an empty array.

    P(empty) = 1 / (1 + average_size), matching Hypothesis st.lists() behavior.
    For max_size=10: average_size=5, P(empty) = 1/6 ≈ 16.7%
    '''
    if min_items > 0:
        return st.just(False)
    if max_items <= 0:
        return st.just(True)
    average_size = max_items // 2
    return st.integers(min_value=0, max_value=average_size).map(lambda x: x == 0)


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
        Maximum number of scalars in the array. For structured dtypes, each
        element counts as multiple scalars (one per field).

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
