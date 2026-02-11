import numpy as np
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import EmptyArray, NumpyArray


def leaf_contents(
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,
    min_size: int = 0,
    max_size: int = 10,
    allow_numpy: bool = True,
    allow_empty: bool = True,
) -> st.SearchStrategy[NumpyArray | EmptyArray]:
    if not allow_numpy and not allow_empty:
        raise ValueError('at least one leaf content type must be allowed')

    options: list[st.SearchStrategy[NumpyArray | EmptyArray]] = []
    if allow_numpy:
        options.append(
            st_ak.contents.numpy_array_contents(
                dtypes, allow_nan, min_size=min_size, max_size=max_size
            )
        )
    if allow_empty and min_size == 0:
        options.append(st_ak.contents.empty_array_contents())
    return st.one_of(options)
