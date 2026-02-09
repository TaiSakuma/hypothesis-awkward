from collections.abc import Callable

import numpy as np
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak

MAX_REGULAR_SIZE = 5
MAX_LIST_LENGTH = 5

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
        content_fns.append(_regular_array_contents)
    if allow_list_offset:
        content_fns.append(_list_offset_array_contents)
    if allow_list:
        content_fns.append(_list_array_contents)

    layout: ak.contents.Content
    if not content_fns or max_size == 0:
        layout = draw(_numpy_array_contents(dtypes, allow_nan, max_size))
    else:
        leaf_st = _BudgetedNumpyArrayContents(dtypes, allow_nan, max_size)

        # Draw nesting depth, then choose a content function for each level.
        depth = draw(st.integers(min_value=0, max_value=max_depth))
        chosen: list[_ContentsFn] = [
            draw(st.sampled_from(content_fns)) for _ in range(depth)
        ]

        layout = draw(leaf_st)
        for fn in reversed(chosen):
            layout = draw(fn(st.just(layout)))

    return ak.Array(layout)


def _numpy_array_contents(
    dtypes: st.SearchStrategy[np.dtype] | None,
    allow_nan: bool,
    max_size: int,
) -> st.SearchStrategy[ak.contents.NumpyArray]:
    '''Base strategy: leaf NumpyArray Content.'''
    return st_ak.numpy_arrays(
        dtype=dtypes,
        allow_structured=False,
        allow_nan=allow_nan,
        max_dims=1,
        max_size=max_size,
    ).map(ak.contents.NumpyArray)


def _BudgetedNumpyArrayContents(
    dtypes: st.SearchStrategy[np.dtype] | None,
    allow_nan: bool,
    max_size: int,
) -> st.SearchStrategy[ak.contents.NumpyArray]:
    '''Leaf strategy with a scalar budget.'''
    remaining = max_size

    @st.composite
    def _contents(draw: st.DrawFn) -> ak.contents.NumpyArray:
        nonlocal remaining
        if remaining == 0:
            raise _BudgetExhausted
        result = draw(_numpy_array_contents(dtypes, allow_nan, remaining))
        remaining -= len(result)
        return result

    return _contents()


class _BudgetExhausted(Exception):
    pass


@st.composite
def _regular_array_contents(
    draw: st.DrawFn,
    contents: st.SearchStrategy[ak.contents.Content],
) -> ak.contents.Content:
    '''Strategy for RegularArray Content wrapping child Content.'''
    content = draw(contents)
    content_len = len(content)
    if content_len == 0:
        size = draw(st.integers(min_value=0, max_value=MAX_REGULAR_SIZE))
        if size == 0:
            zeros_length = draw(st.integers(min_value=0, max_value=MAX_REGULAR_SIZE))
            return ak.contents.RegularArray(content, size=0, zeros_length=zeros_length)
        return ak.contents.RegularArray(content, size=size)
    divisors = [
        d
        for d in range(1, min(content_len + 1, MAX_REGULAR_SIZE + 1))
        if content_len % d == 0
    ]
    size = draw(st.sampled_from(divisors))
    return ak.contents.RegularArray(content, size=size)


@st.composite
def _list_offset_array_contents(
    draw: st.DrawFn,
    contents: st.SearchStrategy[ak.contents.Content],
) -> ak.contents.Content:
    '''Strategy for ListOffsetArray Content wrapping child Content.'''
    content = draw(contents)
    content_len = len(content)
    n = draw(st.integers(min_value=0, max_value=MAX_LIST_LENGTH))
    if n == 0:
        offsets_list = [0]
    elif content_len == 0:
        offsets_list = [0] * (n + 1)
    else:
        splits = sorted(
            draw(
                st.lists(
                    st.integers(min_value=0, max_value=content_len),
                    min_size=n - 1,
                    max_size=n - 1,
                )
            )
        )
        offsets_list = [0, *splits, content_len]
    offsets = np.array(offsets_list, dtype=np.int64)
    return ak.contents.ListOffsetArray(ak.index.Index64(offsets), content)


@st.composite
def _list_array_contents(
    draw: st.DrawFn,
    contents: st.SearchStrategy[ak.contents.Content],
) -> ak.contents.Content:
    '''Strategy for ListArray Content wrapping child Content.'''
    content = draw(contents)
    content_len = len(content)
    n = draw(st.integers(min_value=0, max_value=MAX_LIST_LENGTH))
    if n == 0:
        offsets_list = [0]
    elif content_len == 0:
        offsets_list = [0] * (n + 1)
    else:
        splits = sorted(
            draw(
                st.lists(
                    st.integers(min_value=0, max_value=content_len),
                    min_size=n - 1,
                    max_size=n - 1,
                )
            )
        )
        offsets_list = [0, *splits, content_len]
    offsets = np.array(offsets_list, dtype=np.int64)
    starts = ak.index.Index64(offsets[:-1])
    stops = ak.index.Index64(offsets[1:])
    return ak.contents.ListArray(starts, stops, content)
