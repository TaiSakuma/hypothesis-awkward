from typing import Any, TypedDict, cast

from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, RegularArray
from hypothesis_awkward.util import iter_contents

MAX_REGULAR_SIZE = 5


class RegularArrayContentsKwargs(TypedDict, total=False):
    '''Options for `regular_array_contents()` strategy.'''

    content: st.SearchStrategy[Content] | Content
    max_size: int


@st.composite
def regular_array_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[RegularArrayContentsKwargs]:
    '''Strategy for options for `regular_array_contents()` strategy.'''
    if chain is None:
        chain = st_ak.OptsChain({})
    st_content = chain.register(st_ak.contents.contents())

    kwargs = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'content': st.one_of(
                    st_ak.contents.contents(),
                    st.just(st_content),
                ),
                'max_size': st.integers(min_value=0, max_value=50),
            },
        )
    )

    return chain.extend(cast(RegularArrayContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_regular_array_contents(data: st.DataObject) -> None:
    '''Test that `regular_array_contents()` respects all its options.'''
    opts = data.draw(regular_array_contents_kwargs(), label='opts')
    opts.reset()
    result = data.draw(
        st_ak.contents.regular_array_contents(**opts.kwargs), label='result'
    )

    # Assert the result is always a RegularArray content
    assert isinstance(result, RegularArray)

    # Assert size is within bounds
    max_size = opts.kwargs.get('max_size', MAX_REGULAR_SIZE)
    assert result.size <= max_size

    # Assert size divides content length when size > 0
    if result.size > 0:
        assert len(result.content) % result.size == 0
        assert len(result) == len(result.content) // result.size

    # Assert content
    content = opts.kwargs.get('content', None)
    match content:
        case Content():
            assert result.content is content
        case st_ak.RecordDraws():
            assert len(content.drawn) == 1
            assert result.content is content.drawn[0]


def test_draw_max_size() -> None:
    '''Assert that RegularArray with exactly max_size can be drawn.'''
    max_size = 10
    find(
        st_ak.contents.regular_array_contents(max_size=max_size),
        lambda c: c.size == max_size,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_from_contents_size_zero() -> None:
    '''Assert that RegularArray with size=0 can be drawn from `contents()`.'''

    def _has_regular_size_zero(c: Content) -> bool:
        return any(
            isinstance(n, RegularArray) and n.size == 0 for n in iter_contents(c)
        )

    find(
        st_ak.contents.contents(),
        _has_regular_size_zero,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
