from collections.abc import Callable

import numpy as np
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward.strategies.numpy import numpy_arrays

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
    allow_nan: bool = False,
    allow_regular: bool = True,
    allow_list_offset: bool = True,
    allow_list: bool = True,
    max_size: int = 10,
    max_depth: int = 3,
) -> ak.Array:
    '''Strategy for Awkward Arrays built from direct Content constructors.

    Parameters
    ----------
    dtypes
        A strategy for NumPy dtypes used in leaf ``NumpyArray`` nodes.
        If ``None``, uses ``supported_dtypes()``.
    allow_nan
        Generate potentially ``NaN``/``NaT`` values for relevant dtypes
        if ``True``.
    allow_regular
        Allow wrapping the leaf ``NumpyArray`` in one or more
        ``RegularArray`` layers if ``True``.
    allow_list_offset
        Allow wrapping the leaf ``NumpyArray`` in one or more
        ``ListOffsetArray`` layers if ``True``.
    allow_list
        Allow wrapping the leaf ``NumpyArray`` in one or more
        ``ListArray`` layers if ``True``.
    max_size
        Maximum total number of leaf scalars in the generated array
        (i.e., the sum of ``arr.size`` across all leaf ``NumpyArray``
        nodes).
    max_depth
        Maximum number of nested structural layers (``RegularArray``,
        ``ListOffsetArray``, ``ListArray``) wrapping the leaf ``NumpyArray``.  Only
        effective when at least one structural type is enabled.

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

    effective_max_depth = max_depth if wrappers else 0
    base = _numpy_leaf(dtypes, allow_nan, max_size)
    if effective_max_depth == 0:
        layout = draw(base)
    else:
        max_leaves = 2 ** (effective_max_depth - 1)

        def extend(
            children: st.SearchStrategy[ak.contents.Content],
        ) -> st.SearchStrategy[ak.contents.Content]:
            return st.one_of(*[w(children) for w in wrappers])

        layout = draw(st.recursive(base, extend, max_leaves=max_leaves))
    return ak.Array(layout)


def _numpy_leaf(
    dtypes: st.SearchStrategy[np.dtype] | None,
    allow_nan: bool,
    max_size: int,
) -> st.SearchStrategy[ak.contents.NumpyArray]:
    '''Base strategy: leaf NumpyArray Content.'''
    return numpy_arrays(
        dtype=dtypes,
        allow_structured=False,
        allow_nan=allow_nan,
        max_dims=1,
        max_size=max_size,
    ).map(ak.contents.NumpyArray)


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
            zeros_length = draw(
                st.integers(min_value=0, max_value=MAX_REGULAR_SIZE)
            )
            return ak.contents.RegularArray(
                child, size=0, zeros_length=zeros_length
            )
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
