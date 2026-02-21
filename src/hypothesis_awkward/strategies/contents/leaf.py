import numpy as np
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import EmptyArray, ListOffsetArray, NumpyArray


def leaf_contents(
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,
    min_size: int = 0,
    max_size: int = 10,
    allow_numpy: bool = True,
    allow_empty: bool = True,
    allow_string: bool = True,
    allow_bytestring: bool = True,
) -> st.SearchStrategy[NumpyArray | EmptyArray | ListOffsetArray]:
    '''Strategy for leaf content types.

    Produces one of NumpyArray, EmptyArray, string, or bytestring content,
    selected by ``st.one_of``.

    Parameters
    ----------
    dtypes
        A strategy for NumPy scalar dtypes used in ``NumpyArray``. If ``None``,
        the default strategy that generates any scalar dtype supported by
        Awkward Array is used. Does not affect string or bytestring content.
    allow_nan
        No ``NaN``/``NaT`` values are generated in ``NumpyArray`` if ``False``.
    min_size
        Minimum number of elements.
    max_size
        Maximum number of elements.
    allow_numpy
        No ``NumpyArray`` is generated if ``False``.
    allow_empty
        No ``EmptyArray`` is generated if ``False``.
    allow_string
        No string content is generated if ``False``.
    allow_bytestring
        No bytestring content is generated if ``False``.

    Examples
    --------
    >>> c = leaf_contents().example()
    >>> isinstance(c, (NumpyArray, EmptyArray, ListOffsetArray))
    True
    '''
    if not any((allow_numpy, allow_empty, allow_string, allow_bytestring)):
        raise ValueError('at least one leaf content type must be allowed')

    options: list[st.SearchStrategy[NumpyArray | EmptyArray | ListOffsetArray]] = []
    if allow_numpy:
        options.append(
            st_ak.contents.numpy_array_contents(
                dtypes=dtypes, allow_nan=allow_nan, min_size=min_size, max_size=max_size
            )
        )
    if allow_empty and min_size == 0:
        options.append(st_ak.contents.empty_array_contents())
    if allow_string:
        options.append(
            st_ak.contents.string_contents(min_size=min_size, max_size=max_size)
        )
    if allow_bytestring:
        options.append(
            st_ak.contents.bytestring_contents(min_size=min_size, max_size=max_size)
        )
    return st.one_of(options)
