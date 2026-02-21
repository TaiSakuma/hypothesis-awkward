import functools

import numpy as np
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content
from hypothesis_awkward.strategies.contents.leaf import leaf_contents
from hypothesis_awkward.util.draw import CountdownDrawer


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
    allow_record: bool = True,
    allow_union: bool = True,
    max_depth: int = 5,
) -> Content:
    '''Strategy for Awkward Array content layouts.

    Builds content layouts by recursively constructing a tree of content
    nodes. At each level, a coin flip decides whether to go deeper or
    produce a leaf. Leaf types include NumpyArray, EmptyArray, string,
    and bytestring. Wrapper types include RegularArray, ListOffsetArray,
    ListArray, RecordArray, and UnionArray.

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
    allow_record
        No ``RecordArray`` is generated if ``False``.
    allow_union
        No ``UnionArray`` is generated if ``False``.
    max_depth
        Maximum nesting depth. At each level below this limit, a coin flip
        decides whether to descend further or produce a leaf. Each
        RegularArray, ListOffsetArray, ListArray, RecordArray, and UnionArray
        layer adds one level, excluding those that form string or bytestring
        content.

    Examples
    --------
    >>> c = contents().example()
    >>> isinstance(c, Content)
    True

    '''
    st_leaf = functools.partial(
        leaf_contents,
        dtypes=dtypes,
        allow_nan=allow_nan,
        allow_numpy=allow_numpy,
        allow_empty=allow_empty,
        allow_string=allow_string,
        allow_bytestring=allow_bytestring,
    )

    leaf_only = (
        not any(
            (
                allow_regular,
                allow_list_offset,
                allow_list,
                allow_record,
                allow_union,
            )
        )
        or max_size == 0
    )
    if leaf_only:
        return draw(st_leaf(min_size=0, max_size=max_size))

    draw_leaf = CountdownDrawer(draw, st_leaf, max_size_total=max_size)

    def _build(depth: int, *, allow_union: bool = True) -> Content | None:
        if depth >= max_depth or not draw(st.booleans()):
            return draw_leaf()

        # Choose node type from allow_* flags
        candidates: list[str] = []
        if allow_regular:
            candidates.append('regular')
        if allow_list_offset:
            candidates.append('list_offset')
        if allow_list:
            candidates.append('list')
        if allow_record:
            candidates.append('record')
        if allow_union:
            candidates.append('union')

        if not candidates:
            return draw_leaf()

        node_type = draw(st.sampled_from(sorted(candidates)))

        # Build children for multi-child types
        if node_type == 'union':
            first = _build(depth + 1, allow_union=False)
            if first is None:
                return None
            second = _build(depth + 1, allow_union=False)
            if second is None:
                return None
            children = [first, second]
            while draw(st.booleans()):
                child = _build(depth + 1, allow_union=False)
                if child is None:
                    break
                children.append(child)
            return draw(st_ak.contents.union_array_contents(children))

        if node_type == 'record':
            first = _build(depth + 1, allow_union=allow_union)
            if first is None:
                return None
            children = [first]
            while draw(st.booleans()):
                child = _build(depth + 1, allow_union=allow_union)
                if child is None:
                    break
                children.append(child)
            return draw(st_ak.contents.record_array_contents(children))

        # Single-child wrapper
        child = _build(depth + 1, allow_union=allow_union)
        if child is None:
            return None
        if node_type == 'regular':
            return draw(st_ak.contents.regular_array_contents(child))
        if node_type == 'list_offset':
            return draw(st_ak.contents.list_offset_array_contents(child))
        return draw(st_ak.contents.list_array_contents(child))

    result = _build(0, allow_union=allow_union)
    if result is not None:
        return result
    return draw(st_leaf(min_size=0, max_size=0))
