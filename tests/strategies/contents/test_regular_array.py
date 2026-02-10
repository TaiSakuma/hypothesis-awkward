from typing import TypedDict, cast

from hypothesis import given, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak

MAX_REGULAR_SIZE = 5


class RegularArrayContentsKwargs(TypedDict, total=False):
    '''Options for `regular_array_contents()` strategy.'''

    contents: st.SearchStrategy[ak.contents.Content]


def regular_array_contents_kwargs() -> st.SearchStrategy[
    st_ak.Opts[RegularArrayContentsKwargs]
]:
    '''Strategy for options for `regular_array_contents()` strategy.'''
    return (
        st.fixed_dictionaries(
            {},
            optional={
                'contents': st.just(
                    st_ak.RecordDraws(st_ak.contents.numpy_array_contents())
                ),
            },
        )
        .map(lambda d: cast(RegularArrayContentsKwargs, d))
        .map(st_ak.Opts[RegularArrayContentsKwargs])
    )


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
    assert isinstance(result, ak.contents.RegularArray)

    # Assert size is within bounds
    assert 0 <= result.size <= MAX_REGULAR_SIZE

    # Assert size divides content length when size > 0
    if result.size > 0:
        assert len(result.content) % result.size == 0
        assert len(result) == len(result.content) // result.size

    # Assert contents
    contents = opts.kwargs.get('contents', None)
    match contents:
        case st_ak.RecordDraws():
            assert len(contents.drawn) == 1
            assert result.content is contents.drawn[0]
