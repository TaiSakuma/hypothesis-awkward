import numpy as np
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, ListOffsetArray


@st.composite
def list_offset_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    max_length: int = 5,
) -> Content:
    '''Strategy for ListOffsetArray Content wrapping child Content.

    Parameters
    ----------
    content
        Child content. Can be a strategy for Content, a concrete Content
        instance, or ``None`` to draw from ``contents()``.
    max_length
        Upper bound on the number of lists, i.e., ``len(result)``.

    Examples
    --------
    >>> c = list_offset_array_contents().example()
    >>> isinstance(c, Content)
    True

    Limit the number of lists:

    >>> c = list_offset_array_contents(max_length=4).example()
    >>> len(c) <= 4
    True
    '''
    match content:
        case None:
            content = draw(st_ak.contents.contents())
        case st.SearchStrategy():
            content = draw(content)
        case Content():
            pass
    assert isinstance(content, Content)
    content_len = len(content)
    n = draw(st.integers(min_value=0, max_value=max_length))
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
    return ListOffsetArray(ak.index.Index64(offsets), content)
