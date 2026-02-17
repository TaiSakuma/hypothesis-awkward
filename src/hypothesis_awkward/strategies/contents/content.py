import functools
from collections.abc import Callable

import numpy as np
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content
from hypothesis_awkward.strategies.contents.leaf import leaf_contents
from hypothesis_awkward.util.draw import CountdownDrawer

_WrapperFn = Callable[[st.SearchStrategy[Content]], st.SearchStrategy[Content]]


@st.composite
def contents(
    draw: st.DrawFn,
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    max_size: int = 10,
    allow_nan: bool = False,
    allow_numpy: bool = True,
    allow_empty: bool = True,
    allow_string: bool = True,
    allow_bytestring: bool = True,
    allow_regular: bool = True,
    allow_list_offset: bool = True,
    allow_list: bool = True,
    max_depth: int = 5,
) -> Content:
    '''Strategy for Awkward Array content layouts.

    Builds content layouts by recursively constructing a tree of content
    nodes. At each level, a coin flip decides whether to go deeper or
    produce a leaf. Leaf types include NumpyArray, EmptyArray, string,
    and bytestring. Wrapper types include RegularArray, ListOffsetArray,
    and ListArray.

    Parameters
    ----------
    dtypes
        A strategy for NumPy scalar dtypes used in ``NumpyArray``. If ``None``, the
        default strategy that generates any scalar dtype supported by Awkward Array is
        used. Does not affect string or bytestring content.
    max_size
        Maximum total number of elements in the generated content. Each
        numerical value, including complex and datetime, counts as one. Each
        string and bytestring (not character or byte) counts as one.
    allow_nan
        No ``NaN``/``NaT`` values are generated in ``NumpyArray`` if ``False``.
    allow_numpy
        No ``NumpyArray`` is generated if ``False``.
    allow_empty
        No ``EmptyArray`` is generated if ``False``. ``EmptyArray`` has Awkward
        type ``unknown`` and carries no data. Unlike ``NumpyArray``, it is
        unaffected by ``dtypes`` and ``allow_nan``.
    allow_string
        No string content is generated if ``False``. Strings are represented
        as a ``ListOffsetArray`` wrapping a ``NumpyArray(uint8)``. Each
        string (not character) counts toward ``max_size``. The string
        itself does not count toward ``max_depth``. Unaffected by ``dtypes``
        and ``allow_nan``.
    allow_bytestring
        No bytestring content is generated if ``False``. Bytestrings are
        represented as a ``ListOffsetArray`` wrapping a ``NumpyArray(uint8)``.
        Each bytestring (not byte) counts toward ``max_size``. The
        bytestring itself does not count toward ``max_depth``. Unaffected
        by ``dtypes`` and ``allow_nan``.
    allow_regular
        No ``RegularArray`` is generated if ``False``.
    allow_list_offset
        No ``ListOffsetArray`` is generated if ``False``.
    allow_list
        No ``ListArray`` is generated if ``False``.
    max_depth
        Maximum nesting depth. At each level below this limit, a coin flip
        decides whether to descend further or produce a leaf. Each
        RegularArray, ListOffsetArray, and ListArray layer adds one level,
        excluding those that form string or bytestring content.

    Examples
    --------
    >>> c = contents().example()
    >>> isinstance(c, Content)
    True

    '''
    wrappers: dict[str, _WrapperFn] = {}
    if allow_regular:
        wrappers['regular'] = st_ak.contents.regular_array_contents
    if allow_list_offset:
        wrappers['list_offset'] = st_ak.contents.list_offset_array_contents
    if allow_list:
        wrappers['list'] = st_ak.contents.list_array_contents

    st_leaf = functools.partial(
        leaf_contents,
        dtypes=dtypes,
        allow_nan=allow_nan,
        allow_numpy=allow_numpy,
        allow_empty=allow_empty,
        allow_string=allow_string,
        allow_bytestring=allow_bytestring,
    )

    if not wrappers or max_size == 0:
        return draw(st_leaf(min_size=0, max_size=max_size))

    draw_leaf = CountdownDrawer(draw, st_leaf, max_size_total=max_size)

    def _leaf() -> Content:
        content = draw_leaf()
        if content is not None:
            return content
        return draw(st_leaf(min_size=0, max_size=0))

    def _build(depth: int) -> Content:
        if depth >= max_depth or not draw(st.booleans()):
            return _leaf()
        child = _build(depth + 1)
        node_type = draw(st.sampled_from(sorted(wrappers)))
        return draw(wrappers[node_type](st.just(child)))

    return _build(0)
