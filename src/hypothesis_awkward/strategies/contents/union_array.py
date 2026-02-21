import numpy as np
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
) -> Content:
    '''Strategy for UnionArray Content from a list of child Contents.

    Parameters
    ----------
    contents
        Child contents. Can be a strategy for a list of Content, a concrete
        list, or ``None`` to draw random children.
    max_contents
        Maximum number of child contents when ``contents`` is ``None``.

    Examples
    --------
    >>> c = union_array_contents().example()
    >>> isinstance(c, Content)
    True
    '''
    match contents:
        case None:
            n = draw(st.integers(min_value=2, max_value=max_contents))
            contents = [
                draw(st_ak.contents.contents(allow_union=False)) for _ in range(n)
            ]
        case st.SearchStrategy():
            contents = draw(contents)
        case list():
            pass
    assert isinstance(contents, list)

    # Build compact tags and index arrays
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
    else:
        tags_flat = np.array([], dtype=np.int8)
        index_flat = np.array([], dtype=np.int64)

    return UnionArray(
        tags=ak.index.Index8(tags_flat),
        index=ak.index.Index64(index_flat),
        contents=contents,
    )
