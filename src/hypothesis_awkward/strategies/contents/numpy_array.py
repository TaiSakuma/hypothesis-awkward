import numpy as np
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak


def numpy_array_contents(
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,
    min_size: int = 0,
    max_size: int = 10,
) -> st.SearchStrategy[ak.contents.NumpyArray]:
    '''Strategy for NumpyArray content.'''
    return st_ak.numpy_arrays(
        dtype=dtypes,
        allow_structured=False,
        allow_nan=allow_nan,
        max_dims=1,
        min_size=min_size,
        max_size=max_size,
    ).map(ak.contents.NumpyArray)
