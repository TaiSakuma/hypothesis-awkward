from typing import Any, TypedDict, cast

from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, ListOffsetArray

MAX_LIST_LENGTH = 5


class ListOffsetArrayContentsKwargs(TypedDict, total=False):
    '''Options for `list_offset_array_contents()` strategy.'''

    content: st.SearchStrategy[Content] | Content


@st.composite
def list_offset_array_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[ListOffsetArrayContentsKwargs]:
    '''Strategy for options for `list_offset_array_contents()` strategy.'''
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
            },
        )
    )

    return chain.extend(cast(ListOffsetArrayContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_list_offset_array_contents(data: st.DataObject) -> None:
    '''Test that `list_offset_array_contents()` respects all its options.'''
    opts = data.draw(list_offset_array_contents_kwargs(), label='opts')
    opts.reset()
    result = data.draw(
        st_ak.contents.list_offset_array_contents(**opts.kwargs), label='result'
    )

    # Assert the result is always a ListOffsetArray content
    assert isinstance(result, ListOffsetArray)

    # Assert list length is within bounds
    assert 0 <= len(result) <= MAX_LIST_LENGTH

    # Assert offsets are monotonically non-decreasing
    offsets = result.offsets.data
    for i in range(len(offsets) - 1):
        assert offsets[i] <= offsets[i + 1]

    # Assert first offset is 0 and last offset does not exceed content length
    assert offsets[0] == 0
    assert offsets[-1] <= len(result.content)

    # Assert content
    content = opts.kwargs.get('content', None)
    match content:
        case Content():
            assert result.content is content
        case st_ak.RecordDraws():
            assert len(content.drawn) == 1
            assert result.content is content.drawn[0]


def test_draw_from_contents() -> None:
    '''Assert that ListOffsetArray can be drawn from `contents()`.'''

    def _has_list_offset(c: Content) -> bool:
        stack: list[Content] = [c]
        while stack:
            node = stack.pop()
            if isinstance(node, ListOffsetArray):
                return True
            if hasattr(node, 'content'):
                stack.append(node.content)
        return False

    find(
        st_ak.contents.contents(),
        _has_list_offset,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_from_contents_variable_length() -> None:
    '''Assert that variable-length sublists can be drawn from `contents()`.'''

    def _has_variable_length(c: Content) -> bool:
        node = c
        while hasattr(node, 'content'):
            if isinstance(node, ListOffsetArray) and len(node) >= 2:
                lengths = [len(node[i]) for i in range(len(node))]
                if len(set(lengths)) > 1:
                    return True
            node = node.content
        return False

    find(
        st_ak.contents.contents(),
        _has_variable_length,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_from_contents_empty_sublist() -> None:
    '''Assert that empty sublists can be drawn from `contents()`.'''

    def _has_empty_sublist(c: Content) -> bool:
        node = c
        while hasattr(node, 'content'):
            if isinstance(node, ListOffsetArray):
                for i in range(len(node)):
                    if len(node[i]) == 0:
                        return True
            node = node.content
        return False

    find(
        st_ak.contents.contents(),
        _has_empty_sublist,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
