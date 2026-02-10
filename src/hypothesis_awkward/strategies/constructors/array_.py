from collections.abc import Callable

import numpy as np
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward.strategies.contents.list_array import list_array_contents
from hypothesis_awkward.strategies.contents.list_offset_array import (
    list_offset_array_contents,
)
from hypothesis_awkward.strategies.contents.numpy_array import numpy_array_contents
from hypothesis_awkward.strategies.contents.regular_array import (
    regular_array_contents,
)

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
        draw_content = DrawContent(draw, dtypes, allow_nan, max_size)

        # Draw nesting depth, then choose a content function for each level.
        depth = draw(st.integers(min_value=0, max_value=max_depth))
        chosen: list[_ContentsFn] = [
            draw(st.sampled_from(content_fns)) for _ in range(depth)
        ]

        layout = draw_content()
        for fn in reversed(chosen):
            layout = draw(fn(st.just(layout)))

    return ak.Array(layout)


def DrawContent(
    draw: st.DrawFn,
    dtypes: st.SearchStrategy[np.dtype] | None,
    allow_nan: bool,
    max_size: int,
) -> Callable[[], ak.contents.NumpyArray | None]:
    '''Callable that draws NumpyArray content with a depleting element count.

    Returns a function that, when called, draws a ``NumpyArray`` using the
    provided ``draw`` function and reduces the remaining element count by
    the length of the drawn content. Returns ``None`` when the count
    reaches zero.
    '''
    remaining = max_size

    def _draw_content() -> ak.contents.NumpyArray | None:
        nonlocal remaining
        if remaining == 0:
            return None
        result = draw(numpy_array_contents(dtypes, allow_nan, remaining))
        remaining -= len(result)
        return result

    return _draw_content
