from collections.abc import Callable

import numpy as np
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak

MAX_REGULAR_SIZE = 5
MAX_LIST_LENGTH = 5

ExtendFn = Callable[
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
    wrappers: list[ExtendFn] = []
    if allow_regular:
        wrappers.append(_wrap_regular)
    if allow_list_offset:
        wrappers.append(_wrap_list_offset)
    if allow_list:
        wrappers.append(_wrap_list)

    layout: ak.contents.Content
    if not wrappers or max_size == 0:
        layout = draw(_numpy_leaf(dtypes, allow_nan, max_size))
    else:
        draw_leaf = _make_draw_leaf(draw, dtypes, allow_nan, max_size)

        # Draw nesting depth, then choose a wrapper for each level.
        depth = draw(st.integers(min_value=0, max_value=max_depth))
        chosen_wrappers: list[ExtendFn] = [
            draw(st.sampled_from(wrappers)) for _ in range(depth)
        ]

        layout = draw_leaf()
        for wrapper in reversed(chosen_wrappers):
            layout = draw(wrapper(st.just(layout)))

    return ak.Array(layout)


def _numpy_leaf(
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


def _make_draw_leaf(
    draw: st.DrawFn,
    dtypes: st.SearchStrategy[np.dtype] | None,
    allow_nan: bool,
    max_size: int,
) -> Callable[[], ak.contents.NumpyArray]:
    '''Create a draw_leaf function with a scalar budget.'''
    remaining = max_size

    def draw_leaf() -> ak.contents.NumpyArray:
        nonlocal remaining
        if remaining == 0:
            raise _BudgetExhausted
        arr = draw(st_ak.numpy_arrays(
            dtype=dtypes,
            allow_structured=False,
            allow_nan=allow_nan,
            max_dims=1,
            max_size=remaining,
        ))
        remaining -= arr.size
        return ak.contents.NumpyArray(arr)

    return draw_leaf


class _BudgetExhausted(Exception):
    pass


@st.composite
def _wrap_regular(
    draw: st.DrawFn,
    children: st.SearchStrategy[ak.contents.Content],
) -> ak.contents.Content:
    '''Extend strategy: wrap child Content in a RegularArray.'''
    child = draw(children)
    child_len = len(child)
    if child_len == 0:
        size = draw(st.integers(min_value=0, max_value=MAX_REGULAR_SIZE))
        if size == 0:
            zeros_length = draw(st.integers(min_value=0, max_value=MAX_REGULAR_SIZE))
            return ak.contents.RegularArray(child, size=0, zeros_length=zeros_length)
        return ak.contents.RegularArray(child, size=size)
    divisors = [
        d
        for d in range(1, min(child_len + 1, MAX_REGULAR_SIZE + 1))
        if child_len % d == 0
    ]
    size = draw(st.sampled_from(divisors))
    return ak.contents.RegularArray(child, size=size)


@st.composite
def _wrap_list_offset(
    draw: st.DrawFn,
    children: st.SearchStrategy[ak.contents.Content],
) -> ak.contents.Content:
    '''Extend strategy: wrap child Content in a ListOffsetArray.'''
    child = draw(children)
    child_len = len(child)
    n = draw(st.integers(min_value=0, max_value=MAX_LIST_LENGTH))
    if n == 0:
        offsets_list = [0]
    elif child_len == 0:
        offsets_list = [0] * (n + 1)
    else:
        splits = sorted(
            draw(
                st.lists(
                    st.integers(min_value=0, max_value=child_len),
                    min_size=n - 1,
                    max_size=n - 1,
                )
            )
        )
        offsets_list = [0, *splits, child_len]
    offsets = np.array(offsets_list, dtype=np.int64)
    return ak.contents.ListOffsetArray(ak.index.Index64(offsets), child)


@st.composite
def _wrap_list(
    draw: st.DrawFn,
    children: st.SearchStrategy[ak.contents.Content],
) -> ak.contents.Content:
    '''Extend strategy: wrap child Content in a ListArray.'''
    child = draw(children)
    child_len = len(child)
    n = draw(st.integers(min_value=0, max_value=MAX_LIST_LENGTH))
    if n == 0:
        offsets_list = [0]
    elif child_len == 0:
        offsets_list = [0] * (n + 1)
    else:
        splits = sorted(
            draw(
                st.lists(
                    st.integers(min_value=0, max_value=child_len),
                    min_size=n - 1,
                    max_size=n - 1,
                )
            )
        )
        offsets_list = [0, *splits, child_len]
    offsets = np.array(offsets_list, dtype=np.int64)
    starts = ak.index.Index64(offsets[:-1])
    stops = ak.index.Index64(offsets[1:])
    return ak.contents.ListArray(starts, stops, child)
