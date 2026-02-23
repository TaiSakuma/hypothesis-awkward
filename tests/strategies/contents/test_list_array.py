from typing import Any, TypedDict, cast

from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, ListArray
from hypothesis_awkward.util import iter_contents

MAX_LENGTH = 5


class ListArrayContentsKwargs(TypedDict, total=False):
    '''Options for `list_array_contents()` strategy.'''

    content: st.SearchStrategy[Content] | Content
    max_length: int


@st.composite
def list_array_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[ListArrayContentsKwargs]:
    '''Strategy for options for `list_array_contents()` strategy.'''
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
                'max_length': st.integers(min_value=0, max_value=50),
            },
        )
    )

    return chain.extend(cast(ListArrayContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_list_array_contents(data: st.DataObject) -> None:
    '''Test that `list_array_contents()` respects all its options.'''
    opts = data.draw(list_array_contents_kwargs(), label='opts')
    opts.reset()
    result = data.draw(
        st_ak.contents.list_array_contents(**opts.kwargs), label='result'
    )

    # Assert the result is always a ListArray content
    assert isinstance(result, ListArray)

    # Assert list length is within bounds
    max_length = opts.kwargs.get('max_length', MAX_LENGTH)
    assert len(result) <= max_length

    # Assert starts and stops have the same length
    starts = result.starts.data
    stops = result.stops.data
    assert len(starts) == len(stops)

    # Assert starts[i] <= stops[i] and stops do not exceed content length
    for i in range(len(starts)):
        assert starts[i] <= stops[i]
        assert stops[i] <= len(result.content)

    # Assert content
    content = opts.kwargs.get('content', None)
    match content:
        case Content():
            assert result.content is content
        case st_ak.RecordDraws():
            assert len(content.drawn) == 1
            assert result.content is content.drawn[0]


def test_draw_max_length() -> None:
    '''Assert that max_length constrains the ListArray length.'''
    max_length = 10
    find(
        st_ak.contents.list_array_contents(max_length=max_length),
        lambda c: len(c) == max_length,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_from_contents() -> None:
    '''Assert that ListArray can be drawn from `contents()`.'''

    def _has_list(c: Content) -> bool:
        return any(isinstance(n, ListArray) for n in iter_contents(c))

    find(
        st_ak.contents.contents(),
        _has_list,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_from_contents_variable_length() -> None:
    '''Assert that ListArray with variable-length sublists can be drawn from `contents()`.'''

    def _has_variable_length(c: Content) -> bool:
        return any(
            isinstance(n, ListArray)
            and len(n) >= 2
            and len(set(len(n[i]) for i in range(len(n)))) > 1
            for n in iter_contents(c)
        )

    find(
        st_ak.contents.contents(),
        _has_variable_length,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_from_contents_empty_sublist() -> None:
    '''Assert that ListArray with empty sublists can be drawn from `contents()`.'''

    def _has_empty_sublist(c: Content) -> bool:
        return any(
            isinstance(n, ListArray) and any(len(n[i]) == 0 for i in range(len(n)))
            for n in iter_contents(c)
        )

    find(
        st_ak.contents.contents(),
        _has_empty_sublist,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
