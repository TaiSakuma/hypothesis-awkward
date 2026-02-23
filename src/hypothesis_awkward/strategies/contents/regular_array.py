from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, RegularArray


@st.composite
def regular_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    max_size: int = 5,
) -> Content:
    '''Strategy for RegularArray Content wrapping child Content.

    Parameters
    ----------
    content
        Child content. Can be a strategy for Content, a concrete Content
        instance, or ``None`` to draw from ``contents()``.
    max_size
        Upper bound on the generated RegularArray length.

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
        size = draw(st.integers(min_value=0, max_value=max_size))
        if size == 0:
            zeros_length = draw(st.integers(min_value=0, max_value=max_size))
            return RegularArray(content, size=0, zeros_length=zeros_length)
        return RegularArray(content, size=size)
    divisors = [
        d for d in range(1, min(content_len + 1, max_size + 1)) if content_len % d == 0
    ]
    if not divisors:
        zeros_length = draw(st.integers(min_value=0, max_value=max_size))
        return RegularArray(content, size=0, zeros_length=zeros_length)
    size = draw(st.sampled_from(divisors))
    return RegularArray(content, size=size)
