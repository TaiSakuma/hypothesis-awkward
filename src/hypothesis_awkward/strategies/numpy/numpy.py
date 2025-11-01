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

    shape: tuple[int, ...]
    if max_size <= 0:
        shape = draw(st.just((0,)))
    else:
        max_side = draw(st.integers(min_value=1, max_value=max_size))
        if max_side == 1:
            max_dims = min(NDIM_MAX, max_size)
        else:
            max_dims = min(
                NDIM_MAX, math.floor(math.log(max_size) / math.log(max_side))
            )
        max_dims = draw(st.integers(min_value=1, max_value=max_dims))
        shape = draw(st_np.array_shapes(max_dims=max_dims, max_side=max_side))

    return draw(
        st_np.arrays(dtype=dtype, shape=shape, elements={'allow_nan': allow_nan})
    )


def from_numpy(
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None = None,
    allow_structured: bool = True,
    allow_nan: bool = False,
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
    max_size
        Maximum number of items in the array.

    Examples
    --------
    >>> from_numpy().example()
    <Array ... type='...'>

    '''

    return st.builds(
        ak.from_numpy,
        numpy_arrays(
            dtype=dtype,
            allow_structured=allow_structured,
            allow_nan=allow_nan,
            max_size=max_size,
        ),
    )
