import functools

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, UnionArray


@st.composite
def union_array_contents(
    draw: st.DrawFn,
    contents: list[Content] | st.SearchStrategy[list[Content]] | None = None,
    *,
    max_contents: int = 4,
    max_length: int | None = None,
) -> Content:
    '''Strategy for UnionArray Content from a list of child Contents.

    Parameters
    ----------
    contents
        Child contents. Can be a strategy for a list of Content, a concrete list, or
        ``None`` to draw random children.
    max_contents
        Maximum number of child contents when ``contents`` is ``None``.
    max_length
        Upper bound on the union length, i.e., ``len(result)``. When ``None``, no
        constraint is applied.

    Examples
    --------
    >>> c = union_array_contents().example()
    >>> isinstance(c, Content)
    True

    Limit the union length:

    >>> c = union_array_contents(max_length=4).example()
    >>> len(c) <= 4
    True
    '''
    match contents:
        case None:
            contents = draw(
                st_ak.contents.content_lists(
                    functools.partial(st_ak.contents.contents, allow_union_root=False),
                    max_total_size=max_length if max_length is not None else 10,
                    min_size=2,
                    max_size=max_contents,
                )
            )
            if max_length is not None:
                assume(sum(len(c) for c in contents) <= max_length)
        case st.SearchStrategy():
            contents = draw(contents)
        case list():
            pass
    assert isinstance(contents, list)

    # Build tags and index arrays
    tags_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    for k, content in enumerate(contents):
        length = len(content)
        tags_parts.append(np.full(length, k, dtype=np.int8))
        index_parts.append(np.arange(length, dtype=np.int64))

    if tags_parts:
        tags_flat = np.concatenate(tags_parts)
        index_flat = np.concatenate(index_parts)

        # Shuffle to interleave contents
        perm = draw(st.permutations(range(len(tags_flat))))
        tags_flat = tags_flat[list(perm)]
        index_flat = index_flat[list(perm)]

        # Truncate if concrete/strategy contents exceed max_length
        if max_length is not None and len(tags_flat) > max_length:
            tags_flat = tags_flat[:max_length]
            index_flat = index_flat[:max_length]
    else:
        tags_flat = np.array([], dtype=np.int8)
        index_flat = np.array([], dtype=np.int64)

    return UnionArray(
        tags=ak.index.Index8(tags_flat),
        index=ak.index.Index64(index_flat),
        contents=contents,
    )
