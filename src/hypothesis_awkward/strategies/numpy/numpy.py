import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

import awkward as ak

from .dtype import numpy_dtypes


def numpy_arrays(
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None = None,
    allow_structured: bool = True,
    allow_nan: bool = False,
) -> st.SearchStrategy[np.ndarray]:
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

    Examples
    --------
    >>> n = numpy_arrays().example()
    >>> ak.from_numpy(n)
    <Array ... type='...'>

    '''

    return st_np.arrays(
        dtype=numpy_dtypes(dtype=dtype, allow_array=allow_structured),
        shape=st_np.array_shapes(),
        elements={'allow_nan': allow_nan},
    )


def from_numpy(
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None = None,
    allow_structured: bool = True,
    allow_nan: bool = False,
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

    Examples
    --------
    >>> from_numpy().example()
    <Array ... type='...'>

    '''

    return st.builds(
        ak.from_numpy,
        numpy_arrays(
            dtype=dtype, allow_structured=allow_structured, allow_nan=allow_nan
        ),
    )
