from collections.abc import Callable

import numpy as np
from hypothesis import strategies as st

import awkward as ak

from .list_array import list_array_contents
from .list_offset_array import list_offset_array_contents
from .numpy_array import BudgetedNumpyArrayContents, numpy_array_contents
from .regular_array import regular_array_contents

_ContentsFn = Callable[
    [st.SearchStrategy[ak.contents.Content]],
    st.SearchStrategy[ak.contents.Content],
]


@st.composite
def arrays(
    draw: st.DrawFn,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    max_size: int = 10,
    allow_nan: bool = False,
    allow_regular: bool = True,
    allow_list_offset: bool = True,
    allow_list: bool = True,
    max_depth: int = 5,
) -> ak.Array:
    '''Strategy for Awkward Arrays.

    The current implementation generates arrays with NumpyArray as leaf contents that can
    be nested multiple levels deep in RegularArray, ListOffsetArray, and ListArray lists.

    Parameters
    ----------
    dtypes
        A strategy for NumPy scalar dtypes used in ``NumpyArray``. If ``None``, the
        default strategy that generates any scalar dtype supported by Awkward Array is
        used.
    max_size
        Maximum total number of scalar values in the generated array.
    allow_nan
        No ``NaN``/``NaT`` values are generated if ``False``.
    allow_regular
        No ``RegularArray`` is generated if ``False``.
    allow_list_offset
        No ``ListOffsetArray`` is generated if ``False``.
    allow_list
        No ``ListArray`` is generated if ``False``.
    max_depth
        Maximum depth of nested arrays.

    Examples
    --------
    >>> arrays().example()
    <Array ... type='...'>

    '''
    content_fns: list[_ContentsFn] = []
    if allow_regular:
        content_fns.append(regular_array_contents)
    if allow_list_offset:
        content_fns.append(list_offset_array_contents)
    if allow_list:
        content_fns.append(list_array_contents)

    layout: ak.contents.Content
    if not content_fns or max_size == 0:
        layout = draw(numpy_array_contents(dtypes, allow_nan, max_size))
    else:
        leaf_st = BudgetedNumpyArrayContents(dtypes, allow_nan, max_size)

        # Draw nesting depth, then choose a content function for each level.
        depth = draw(st.integers(min_value=0, max_value=max_depth))
        chosen: list[_ContentsFn] = [
            draw(st.sampled_from(content_fns)) for _ in range(depth)
        ]

        layout = draw(leaf_st)
        for fn in reversed(chosen):
            layout = draw(fn(st.just(layout)))

    return ak.Array(layout)
