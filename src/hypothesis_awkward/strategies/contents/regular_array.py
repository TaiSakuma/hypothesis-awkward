from __future__ import annotations

from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak

MAX_REGULAR_SIZE = 5


@st.composite
def regular_array_contents(
    draw: st.DrawFn,
    contents: st.SearchStrategy[ak.contents.Content] | None = None,
) -> ak.contents.Content:
    '''Strategy for RegularArray Content wrapping child Content.'''
    if contents is None:
        contents = st_ak.contents.numpy_array_contents()
    content = draw(contents)
    content_len = len(content)
    if content_len == 0:
        size = draw(st.integers(min_value=0, max_value=MAX_REGULAR_SIZE))
        if size == 0:
            zeros_length = draw(st.integers(min_value=0, max_value=MAX_REGULAR_SIZE))
            return ak.contents.RegularArray(content, size=0, zeros_length=zeros_length)
        return ak.contents.RegularArray(content, size=size)
    divisors = [
        d
        for d in range(1, min(content_len + 1, MAX_REGULAR_SIZE + 1))
        if content_len % d == 0
    ]
    size = draw(st.sampled_from(divisors))
    return ak.contents.RegularArray(content, size=size)
