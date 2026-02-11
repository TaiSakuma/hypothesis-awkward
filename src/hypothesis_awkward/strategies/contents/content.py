import functools
from collections.abc import Callable

import numpy as np
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, EmptyArray, NumpyArray
from hypothesis_awkward.util.draw import CountdownDrawer

_NestingFn = Callable[[st.SearchStrategy[Content]], st.SearchStrategy[Content]]


@st.composite
def contents(
    draw: st.DrawFn,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    max_size: int = 10,
    allow_nan: bool = False,
    allow_numpy: bool = True,
    allow_empty: bool = True,
    allow_regular: bool = True,
    allow_list_offset: bool = True,
    allow_list: bool = True,
    max_depth: int = 5,
) -> Content:
    '''Strategy for Awkward Array content layouts.

    The current implementation generates layouts with NumpyArray as leaf contents that can
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
    allow_numpy
        No ``NumpyArray`` is generated if ``False``.
    allow_empty
        No ``EmptyArray`` is generated if ``False``. ``EmptyArray`` has Awkward
        type ``unknown`` and carries no data. Unlike ``NumpyArray``, it is
        unaffected by ``dtypes`` and ``allow_nan``.
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
    >>> c = contents().example()
    >>> isinstance(c, Content)
    True

    '''
    nesting_fns: list[_NestingFn] = []
    if allow_regular:
        nesting_fns.append(st_ak.contents.regular_array_contents)
    if allow_list_offset:
        nesting_fns.append(st_ak.contents.list_offset_array_contents)
    if allow_list:
        nesting_fns.append(st_ak.contents.list_array_contents)

    st_leaf = functools.partial(
        _st_leaf,
        dtypes=dtypes,
        allow_nan=allow_nan,
        allow_numpy=allow_numpy,
        allow_empty=allow_empty,
    )

    if not nesting_fns or max_size == 0:
        return draw(st_leaf(min_size=0, max_size=max_size))

    draw_leaf = CountdownDrawer(draw, st_leaf, max_size_total=max_size)
    content = draw_leaf()
    if content is None:
        return draw(st_leaf(min_size=0, max_size=0))

    # Draw nesting depth, then choose a nesting function for each level.
    depth = draw(st.integers(min_value=0, max_value=max_depth))
    nesting: list[_NestingFn] = [
        draw(st.sampled_from(nesting_fns)) for _ in range(depth)
    ]

    for fn in reversed(nesting):
        content = draw(fn(st.just(content)))

    return content


def _st_leaf(
    *,
    dtypes: st.SearchStrategy[np.dtype] | None,
    allow_nan: bool,
    allow_numpy: bool,
    allow_empty: bool,
    min_size: int,
    max_size: int,
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
