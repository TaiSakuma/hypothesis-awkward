from typing import Any, TypedDict, cast

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, UnionArray
from hypothesis_awkward.util import iter_contents

DEFAULT_MAX_CONTENTS = 4


class UnionArrayContentsKwargs(TypedDict, total=False):
    '''Options for `union_array_contents()` strategy.'''

    contents: list[Content] | st.SearchStrategy[list[Content]]
    max_contents: int


@st.composite
def _contents_list(
    draw: st.DrawFn,
) -> list[Content]:
    '''Draw a list of 2..5 Content objects for testing.'''
    n = draw(st.integers(min_value=2, max_value=5))
    return [
        draw(st_ak.contents.contents(max_size=5, max_depth=2, allow_union=False))
        for _ in range(n)
    ]


@st.composite
def union_array_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[UnionArrayContentsKwargs]:
    '''Strategy for options for `union_array_contents()` strategy.'''
    if chain is None:
        chain = st_ak.OptsChain({})
    st_contents = chain.register(_contents_list())

    kwargs = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'contents': st.one_of(
                    _contents_list(),
                    st.just(st_contents),
                ),
                'max_contents': st.integers(min_value=2, max_value=10),
            },
        )
    )

    return chain.extend(cast(UnionArrayContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_union_array_contents(data: st.DataObject) -> None:
    '''Test that `union_array_contents()` respects all its options.'''
    opts = data.draw(union_array_contents_kwargs(), label='opts')
    opts.reset()
    result = data.draw(
        st_ak.contents.union_array_contents(**opts.kwargs), label='result'
    )

    # Assert the result is always a UnionArray
    assert isinstance(result, UnionArray)

    max_contents = opts.kwargs.get('max_contents', DEFAULT_MAX_CONTENTS)
    contents = opts.kwargs.get('contents', None)

    # Contents dispatch
    match contents:
        case None:
            assert len(result.contents) <= max_contents
        case list():
            # Concrete list: contents should be the same objects
            assert len(result.contents) == len(contents)
            for r_content, given_content in zip(result.contents, contents):
                assert r_content is given_content
        case st_ak.RecordDraws():
            assert len(contents.drawn) == 1
            drawn_list = contents.drawn[0]
            assert len(result.contents) == len(drawn_list)
            for r_content, drawn_content in zip(result.contents, drawn_list):
                assert r_content is drawn_content

    # Tags validity: dtype is int8, values in [0, len(contents))
    tags = result.tags.data
    assert tags.dtype == np.int8
    assert np.all(tags >= 0)
    assert np.all(tags < len(result.contents))

    # Index validity: dtype in {int32, uint32, int64}, values in valid range
    index = result.index.data
    assert index.dtype in (np.int32, np.uint32, np.int64)
    for tag_val in range(len(result.contents)):
        content_len = len(result.contents[tag_val])
        indices_for_tag = index[tags == tag_val]
        if len(indices_for_tag) > 0:
            assert np.all(indices_for_tag >= 0)
            assert np.all(indices_for_tag < content_len)

    # Length: tags and index have the same length
    assert len(tags) == len(index)
    assert len(result) == len(tags)

    # Compact indexing: union length equals sum of content lengths,
    # and each content element is referenced exactly once
    assert len(result) == sum(len(c) for c in result.contents)
    for tag_val in range(len(result.contents)):
        indices_for_tag = index[tags == tag_val]
        content_len = len(result.contents[tag_val])
        assert len(indices_for_tag) == content_len
        assert set(indices_for_tag.tolist()) == set(range(content_len))


def test_draw_multiple_contents() -> None:
    '''Assert that a union with max_contents contents can be drawn.'''
    max_contents = 4
    find(
        st_ak.contents.union_array_contents(max_contents=max_contents),
        lambda u: len(u.contents) == max_contents,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_different_content_lengths() -> None:
    '''Assert that a union with different-length contents can be drawn.'''
    find(
        st_ak.contents.union_array_contents(),
        lambda u: len({len(c) for c in u.contents}) > 1,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_from_contents() -> None:
    '''Assert that UnionArray can be drawn from `contents()`.'''
    find(
        st_ak.contents.contents(max_size=20),
        lambda c: isinstance(c, UnionArray)
        or any(isinstance(n, UnionArray) for n in iter_contents(c)),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
