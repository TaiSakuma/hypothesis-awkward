import functools
import sys
from typing import Literal, Protocol

if sys.version_info >= (3, 11):
    from typing import assert_never
else:
    from typing_extensions import assert_never

import numpy as np
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content
from hypothesis_awkward.strategies.contents.leaf import leaf_contents
from hypothesis_awkward.util.awkward import iter_leaf_contents
from hypothesis_awkward.util.safe import safe_compare as sc

_NodeType = Literal['list', 'list_offset', 'record', 'regular', 'union']


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
    allow_union_root: bool = True,
    max_depth: int = 5,
    max_length: int | None = None,
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
    allow_union_root
        The outermost content node cannot be a ``UnionArray`` if ``False``.
        Unlike ``allow_union``, this does not prevent ``UnionArray`` at
        deeper levels. Awkward Array does not allow a ``UnionArray`` to
        directly contain another ``UnionArray``.
    max_depth
        Maximum nesting depth. At each level below this limit, a coin flip
        decides whether to descend further or produce a leaf. Each
        RegularArray, ListOffsetArray, ListArray, RecordArray, and UnionArray
        layer adds one level, excluding those that form string or bytestring
        content.
    max_length
        Maximum ``len()`` of the generated content. No constraint when ``None`` (the
        default).

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
    leaf_max_size = min(max_size, max_length) if max_length is not None else max_size

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
        return draw(st_leaf(min_size=0, max_size=leaf_max_size))

    if max_depth <= 0 or not draw(st.booleans()):
        return draw(st_leaf(min_size=0, max_size=leaf_max_size))

    recurse = functools.partial(
        contents,
        dtypes=dtypes,
        allow_nan=allow_nan,
        allow_numpy=allow_numpy,
        allow_empty=allow_empty,
        allow_string=allow_string,
        allow_bytestring=allow_bytestring,
        max_depth=max_depth - 1,
        allow_regular=allow_regular,
        allow_list_offset=allow_list_offset,
        allow_list=allow_list,
        allow_record=allow_record,
        allow_union=allow_union,
    )

    # Choose node type from allow_* flags
    candidates = list[_NodeType]()
    if allow_regular:
        candidates.append('regular')
    if allow_list_offset:
        candidates.append('list_offset')
    if allow_list:
        candidates.append('list')
    if allow_record:
        candidates.append('record')
    if allow_union and allow_union_root:
        candidates.append('union')

    if not candidates:
        return draw(st_leaf(min_size=0, max_size=leaf_max_size))

    node_type = draw(st.sampled_from(sorted(candidates)))

    match node_type:
        case 'union':
            children = draw(
                content_lists(
                    functools.partial(recurse, allow_union_root=False),
                    max_total_size=max_size,
                    min_size=2,
                )
            )
            return draw(
                st_ak.contents.union_array_contents(children, max_length=max_length)
            )

        case 'record':
            children = draw(content_lists(recurse, max_total_size=max_size, min_size=1))
            return draw(
                st_ak.contents.record_array_contents(children, max_length=max_length)
            )

        case 'regular':
            child = draw(recurse(max_size=max_size))
            return draw(
                st_ak.contents.regular_array_contents(child, max_length=max_length)
            )

        case 'list_offset':
            child = draw(recurse(max_size=max_size))
            if max_length is not None:
                return draw(
                    st_ak.contents.list_offset_array_contents(
                        child, max_length=max_length
                    )
                )
            return draw(st_ak.contents.list_offset_array_contents(child))

        case 'list':
            child = draw(recurse(max_size=max_size))
            if max_length is not None:
                return draw(
                    st_ak.contents.list_array_contents(child, max_length=max_length)
                )
            return draw(st_ak.contents.list_array_contents(child))

        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


class _StContent(Protocol):
    def __call__(self, *, max_size: int) -> st.SearchStrategy[Content]: ...


@st.composite
def content_lists(
    draw: st.DrawFn,
    st_content: _StContent = contents,
    *,
    max_total_size: int = 10,
    min_size: int = 0,
    max_size: int | None = None,
) -> list[Content]:
    '''Strategy for lists of contents within a size budget.

    Parameters
    ----------
    st_content
        A callable that accepts ``max_size`` and returns a strategy for
        a single content.
    max_total_size
        Maximum total number of leaf elements across all contents in the
        list.
    min_size
        Minimum number of contents in the list.
    max_size
        Maximum number of contents in the list. By default there is no
        upper bound.

    '''
    remaining = max_total_size
    contents_ = list[Content]()
    for _ in range(min_size):
        c = draw(st_content(max_size=max(remaining, 0)))
        remaining -= _leaf_size(c)
        contents_.append(c)
    while draw(st.booleans()) and remaining > 0 and len(contents_) < sc(max_size):
        c = draw(st_content(max_size=max(remaining, 0)))
        remaining -= _leaf_size(c)
        contents_.append(c)
    return contents_


def _leaf_size(c: Content) -> int:
    '''Count total leaf elements in a content tree.'''
    return sum(len(leaf) for leaf in iter_leaf_contents(c))
