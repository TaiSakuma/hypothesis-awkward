import functools
import string
from collections.abc import Callable

import numpy as np
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, RecordArray, UnionArray
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
    wrappers: dict[str, _WrapperFn] = {}
    if allow_regular:
        wrappers['regular'] = st_ak.contents.regular_array_contents
    if allow_list_offset:
        wrappers['list_offset'] = st_ak.contents.list_offset_array_contents
    if allow_list:
        wrappers['list'] = st_ak.contents.list_array_contents

    can_branch = allow_record or allow_union

    single_child_types = list(wrappers)
    if allow_record:
        single_child_types.append('record')

    st_leaf = functools.partial(
        leaf_contents,
        dtypes=dtypes,
        allow_nan=allow_nan,
        allow_numpy=allow_numpy,
        allow_empty=allow_empty,
        allow_string=allow_string,
        allow_bytestring=allow_bytestring,
    )

    if (not single_child_types and not can_branch) or max_size == 0:
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

        # Go down first edge
        children = [_build(depth + 1)]

        # Going up: another edge?
        while can_branch and draw(st.booleans()):
            children.append(_build(depth + 1))

        # Draw node type constrained by child count
        if len(children) == 1:
            if not single_child_types:
                return children[0]
            node_type = draw(st.sampled_from(sorted(single_child_types)))
        else:
            multi_child_types: list[str] = []
            if allow_record:
                multi_child_types.append('record')
            if allow_union and not any(isinstance(c, UnionArray) for c in children):
                multi_child_types.append('union')
            if not multi_child_types:
                # Neither record nor union allowed; fall back to wrapping first child
                if single_child_types:
                    node_type = draw(st.sampled_from(sorted(single_child_types)))
                    return draw(wrappers[node_type](st.just(children[0])))
                return children[0]
            node_type = draw(st.sampled_from(sorted(multi_child_types)))

        # Construct node
        if node_type == 'union':
            return draw(st_ak.contents.union_array_contents(children))

        if node_type == 'record':
            is_tuple = draw(st.booleans())
            if is_tuple:
                fields = None
            else:
                st_names = st.text(
                    alphabet=string.ascii_letters, max_size=3
                )
                fields = draw(
                    st.lists(
                        st_names,
                        min_size=len(children),
                        max_size=len(children),
                        unique=True,
                    )
                )
            length = min(len(c) for c in children)
            return RecordArray(children, fields=fields, length=length)

        # Single-child wrapper
        return draw(wrappers[node_type](st.just(children[0])))

    return _build(0)
