import numpy as np
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import NumpyArray


def numpy_array_contents(
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,
    min_size: int = 0,
    max_size: int = 10,
) -> st.SearchStrategy[NumpyArray]:
    '''Strategy for NumpyArray content.

    Parameters
    ----------
    dtypes
        A strategy for NumPy scalar dtypes. If ``None``, the default strategy that
        generates any scalar dtype supported by Awkward Array is used.
    allow_nan
        No ``NaN``/``NaT`` values are generated if ``False``.
    min_size
        Minimum number of elements.
    max_size
        Maximum number of elements.

    Examples
    --------
    >>> c = numpy_array_contents().example()
    >>> isinstance(c, NumpyArray)
    True
    '''
    return st_ak.numpy_arrays(
        dtype=dtypes,
        allow_structured=False,
        allow_nan=allow_nan,
        max_dims=1,
        min_size=min_size,
        max_size=max_size,
    ).map(NumpyArray)
