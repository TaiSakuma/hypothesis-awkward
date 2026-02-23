from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, RegularArray


@st.composite
def regular_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    max_size: int = 5,
    max_zeros_length: int = 5,
) -> Content:
    '''Strategy for RegularArray Content wrapping child Content.

    Parameters
    ----------
    content
        Child content. Can be a strategy for Content, a concrete Content
        instance, or ``None`` to draw from ``contents()``.
    max_size
        Upper bound on the length of each element.
    max_zeros_length
        Upper bound on the number of elements when each element is
        empty, i.e., when size is zero.

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
    size = draw(_st_group_sizes(len(content), max_size))
    if size == 0:
        zeros_length = draw(st.integers(min_value=0, max_value=max_zeros_length))
        return RegularArray(content, size=0, zeros_length=zeros_length)
    return RegularArray(content, size=size)


def _st_group_sizes(total_items: int, max_group_size: int) -> st.SearchStrategy[int]:
    '''Strategy for the size parameter of a RegularArray.

    A RegularArray subdivides ``total_items`` into equal groups of
    ``group_size``, so ``group_size`` must be a divisor of
    ``total_items`` and at most ``max_group_size``.

    When ``total_items == 0``, any group size up to ``max_group_size``
    is valid because zero items can be split into zero groups of any
    size.

    When ``total_items > 0`` but no valid divisor exists (i.e.,
    ``max_group_size == 0``), returns ``0``. The caller uses this to
    fall back to the ``zeros_length`` path, producing a RegularArray
    whose elements are all empty.
    '''
    if total_items == 0:
        return st.integers(min_value=0, max_value=max_group_size)
    max_group_size = min(total_items, max_group_size)
    group_sizes = [d for d in range(1, max_group_size + 1) if total_items % d == 0]
    if not group_sizes:
        return st.just(0)
    return st.sampled_from(group_sizes)
