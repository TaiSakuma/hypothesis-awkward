from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, RegularArray

MAX_REGULAR_SIZE = 5


@st.composite
def regular_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
) -> Content:
    '''Strategy for RegularArray Content wrapping child Content.

    Parameters
    ----------
    content
        Child content. Can be a strategy for Content, a concrete Content
        instance, or ``None`` to draw from ``contents()``.

    Examples
    --------
    >>> c = regular_array_contents().example()
    >>> isinstance(c, Content)
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
    if content_len == 0:
        size = draw(st.integers(min_value=0, max_value=MAX_REGULAR_SIZE))
        if size == 0:
            zeros_length = draw(st.integers(min_value=0, max_value=MAX_REGULAR_SIZE))
            return RegularArray(content, size=0, zeros_length=zeros_length)
        return RegularArray(content, size=size)
    divisors = [
        d
        for d in range(1, min(content_len + 1, MAX_REGULAR_SIZE + 1))
        if content_len % d == 0
    ]
    size = draw(st.sampled_from(divisors))
    return RegularArray(content, size=size)
