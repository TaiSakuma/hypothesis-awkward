from typing import Any, TypedDict, cast

from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, RecordArray
from hypothesis_awkward.util import iter_contents

DEFAULT_MAX_FIELDS = 5


class RecordArrayContentsKwargs(TypedDict, total=False):
    '''Options for `record_array_contents()` strategy.'''

    contents: list[Content] | st.SearchStrategy[list[Content]]
    max_fields: int
    allow_tuple: bool


@st.composite
def _contents_list(
    draw: st.DrawFn,
) -> list[Content]:
    '''Draw a list of 1..5 Content objects for testing.'''
    n = draw(st.integers(min_value=1, max_value=5))
    return [draw(st_ak.contents.contents(max_size=5, max_depth=2)) for _ in range(n)]


@st.composite
def record_array_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[RecordArrayContentsKwargs]:
    '''Strategy for options for `record_array_contents()` strategy.'''
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
                'max_fields': st.integers(min_value=0, max_value=10),
                'allow_tuple': st.booleans(),
            },
        )
    )

    return chain.extend(cast(RecordArrayContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_record_array_contents(data: st.DataObject) -> None:
    '''Test that `record_array_contents()` respects all its options.'''
    opts = data.draw(record_array_contents_kwargs(), label='opts')
    opts.reset()
    result = data.draw(
        st_ak.contents.record_array_contents(**opts.kwargs), label='result'
    )

    # Assert the result is always a RecordArray
    assert isinstance(result, RecordArray)

    max_fields = opts.kwargs.get('max_fields', DEFAULT_MAX_FIELDS)
    allow_tuple = opts.kwargs.get('allow_tuple', True)
    contents = opts.kwargs.get('contents', None)

    # When auto-generating contents, field count is bounded by max_fields
    match contents:
        case None:
            assert len(result.contents) <= max_fields
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

    # allow_tuple=False means the result must be a named record
    if not allow_tuple:
        assert not result.is_tuple

    # Named records have unique field names matching the number of contents
    if not result.is_tuple:
        assert len(result.fields) == len(result.contents)
        assert len(set(result.fields)) == len(result.fields)

    # Non-empty records: length equals the minimum content length
    if result.contents:
        assert result.length == min(len(c) for c in result.contents)


def test_draw_tuple() -> None:
    '''Assert that `record_array_contents()` can produce a tuple record.'''
    find(
        st_ak.contents.record_array_contents(),
        lambda r: r.is_tuple,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_named() -> None:
    '''Assert that `record_array_contents()` can produce a named record.'''
    find(
        st_ak.contents.record_array_contents(),
        lambda r: not r.is_tuple,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_max_fields() -> None:
    '''Assert that `record_array_contents()` can produce a record with max_fields fields.'''
    max_fields = 3
    find(
        st_ak.contents.record_array_contents(max_fields=max_fields),
        lambda r: len(r.contents) == max_fields,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_from_contents() -> None:
    '''Assert that RecordArray can be drawn from `contents()`.'''
    find(
        st_ak.contents.contents(max_size=20),
        lambda c: any(isinstance(n, RecordArray) for n in iter_contents(c)),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
