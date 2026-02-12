from typing import Any, TypedDict, cast

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import NumpyArray
from hypothesis_awkward.util import (
    any_nan_in_numpy_array,
    any_nan_nat_in_numpy_array,
    any_nat_in_numpy_array,
)

DEFAULT_MAX_SIZE = 10


class NumpyArrayContentsKwargs(TypedDict, total=False):
    '''Options for `numpy_array_contents()` strategy.'''

    dtypes: st.SearchStrategy[np.dtype] | None
    allow_nan: bool
    min_size: int
    max_size: int


@st.composite
def numpy_array_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[NumpyArrayContentsKwargs]:
    '''Strategy for options for `numpy_array_contents()` strategy.'''
    if chain is None:
        chain = st_ak.OptsChain({})
    st_dtypes = chain.register(st_ak.supported_dtypes())

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
                'dtypes': st.one_of(
                    st.none(),
                    st.just(st_dtypes),
                ),
                'allow_nan': st.booleans(),
            },
        )
    )

    return chain.extend(cast(NumpyArrayContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_numpy_array_contents(data: st.DataObject) -> None:
    '''Test that `numpy_array_contents()` respects all its options.'''
    # Draw options
    opts = data.draw(numpy_array_contents_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    result = data.draw(
        st_ak.contents.numpy_array_contents(**opts.kwargs), label='result'
    )

    # Assert the result is always a NumpyArray content
    assert isinstance(result, ak.contents.NumpyArray)

    # Assert underlying data is 1-D (from max_dims=1)
    assert result.data.ndim == 1

    # Assert not structured (from allow_structured=False)
    assert result.data.dtype.names is None

    # Assert size bounds
    dtypes = opts.kwargs.get('dtypes', None)
    allow_nan = opts.kwargs.get('allow_nan', False)
    min_size = opts.kwargs.get('min_size', 0)
    max_size = opts.kwargs.get('max_size', DEFAULT_MAX_SIZE)

    assert min_size <= len(result) <= max_size

    # Assert allow_nan
    if not allow_nan:
        assert not any_nan_nat_in_numpy_array(result.data)

    # Assert dtypes
    match dtypes:
        case st_ak.RecordDraws():
            drawn_kinds = {d.kind for d in dtypes.drawn}
            assert result.data.dtype.kind in drawn_kinds


def test_draw_from_contents() -> None:
    '''Assert that NumpyArray can be drawn from `contents()`.'''
    find(
        st_ak.contents.contents(),
        lambda c: isinstance(c, NumpyArray),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_from_contents_length_zero() -> None:
    '''Assert that NumpyArray with length 0 can be drawn from `contents()`.'''
    find(
        st_ak.contents.contents(),
        lambda c: isinstance(c, NumpyArray) and len(c) == 0,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_empty() -> None:
    '''Assert that empty arrays can be drawn.'''
    find(
        st_ak.contents.numpy_array_contents(),
        lambda c: len(c) == 0,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_max_size() -> None:
    '''Assert that arrays with exactly max_size elements can be drawn.'''
    find(
        st_ak.contents.numpy_array_contents(),
        lambda c: len(c) == DEFAULT_MAX_SIZE,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_min_size() -> None:
    '''Assert that arrays with exactly min_size elements can be drawn.'''
    min_size = 5
    find(
        st_ak.contents.numpy_array_contents(min_size=min_size),
        lambda c: len(c) == min_size,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_nan() -> None:
    '''Assert that arrays with NaN can be drawn when allowed.'''
    float_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'f')
    find(
        st_ak.contents.numpy_array_contents(
            dtypes=float_dtypes, allow_nan=True, min_size=1
        ),
        lambda c: any_nan_in_numpy_array(c.data),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_nat() -> None:
    '''Assert that arrays with NaT can be drawn when allowed.'''
    datetime_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind in ('M', 'm'))
    find(
        st_ak.contents.numpy_array_contents(
            dtypes=datetime_dtypes, allow_nan=True, min_size=1
        ),
        lambda c: any_nat_in_numpy_array(c.data),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
