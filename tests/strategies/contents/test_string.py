from typing import Any, TypedDict, cast

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import ListOffsetArray, NumpyArray
from hypothesis_awkward.util.safe import safe_compare as sc

DEFAULT_MAX_SIZE = 10


class StringContentsKwargs(TypedDict, total=False):
    '''Options for `string_contents()` strategy.'''

    alphabet: st.SearchStrategy[str] | None
    min_size: int
    max_size: int


@st.composite
def string_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[StringContentsKwargs]:
    '''Strategy for options for `string_contents()` strategy.'''
    if chain is None:
        chain = st_ak.OptsChain({})
    st_alphabet = chain.register(st.characters(codec='utf-8'))

    min_size, max_size = draw(
        st_ak.ranges(min_start=0, max_start=DEFAULT_MAX_SIZE, max_end=100)
    )

    drawn = (
        ('min_size', min_size),
        ('max_size', max_size),
    )

    kwargs = draw(
        st.fixed_dictionaries(
            {k: st.just(v) for k, v in drawn if v is not None},
            optional={
                'alphabet': st.one_of(
                    st.none(),
                    st.just(st_alphabet),
                ),
            },
        )
    )

    return chain.extend(cast(StringContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_string_contents(data: st.DataObject) -> None:
    '''Test that `string_contents()` respects all its options.'''
    # Draw options
    opts = data.draw(string_contents_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    result = data.draw(st_ak.contents.string_contents(**opts.kwargs), label='result')

    # Assert the result is always a ListOffsetArray
    assert isinstance(result, ListOffsetArray)

    # Assert outer __array__ parameter
    assert result.parameter('__array__') == 'string'

    # Assert inner content is NumpyArray
    assert isinstance(result.content, NumpyArray)

    # Assert inner __array__ parameter
    assert result.content.parameter('__array__') == 'char'

    # Assert inner dtype is uint8
    assert result.content.dtype == np.dtype(np.uint8)

    # Assert size bounds
    alphabet = opts.kwargs.get('alphabet', None)
    min_size = opts.kwargs.get('min_size', 0)
    max_size = opts.kwargs.get('max_size', DEFAULT_MAX_SIZE)

    assert sc(min_size) <= len(result) <= sc(max_size)

    # Assert all strings are valid UTF-8
    arr = ak.Array(result)
    for s in arr.to_list():
        assert isinstance(s, str)
        s.encode('utf-8')

    # Assert alphabet constraint
    match alphabet:
        case st_ak.RecordDraws():
            drawn_chars = set(alphabet.drawn)
            for s in arr.to_list():
                for ch in s:
                    assert ch in drawn_chars


def test_draw_empty() -> None:
    '''Assert that empty arrays (zero strings) can be drawn.'''
    find(
        st_ak.contents.string_contents(),
        lambda c: len(c) == 0,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_max_size() -> None:
    '''Assert that arrays with exactly max_size strings can be drawn.'''
    find(
        st_ak.contents.string_contents(),
        lambda c: len(c) == DEFAULT_MAX_SIZE,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_empty_string() -> None:
    '''Assert that arrays containing an empty string can be drawn.'''
    find(
        st_ak.contents.string_contents(min_size=1),
        lambda c: '' in ak.Array(c).to_list(),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_non_ascii() -> None:
    '''Assert that arrays containing non-ASCII characters can be drawn.'''
    find(
        st_ak.contents.string_contents(min_size=1),
        lambda c: any(not s.isascii() for s in ak.Array(c).to_list()),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
