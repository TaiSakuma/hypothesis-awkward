from typing import TypedDict, cast

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util import any_nan_in_numpy_array, any_nan_nat_in_numpy_array


class NumpyArrayContentsKwargs(TypedDict):
    '''Options for `numpy_array_contents()` strategy.'''

    dtypes: st.SearchStrategy[np.dtype] | None
    allow_nan: bool
    min_size: int
    max_size: int


@st.composite
def numpy_array_contents_kwargs(
    draw: st.DrawFn,
) -> st_ak.Opts[NumpyArrayContentsKwargs]:
    '''Strategy for options for `numpy_array_contents()` strategy.'''

    min_size, max_size = draw(
        st_ak.ranges(
            min_start=0,
            max_end=100,
            allow_start_none=False,
            allow_end_none=False,
        )
    )

    kwargs = draw(
        st.fixed_dictionaries(
            {
                'min_size': st.just(min_size),
                'max_size': st.just(max_size),
                'dtypes': st.one_of(
                    st.none(),
                    st.just(st_ak.RecordDraws(st_ak.supported_dtypes())),
                ),
                'allow_nan': st.booleans(),
            },
        )
    )

    return st_ak.Opts(cast(NumpyArrayContentsKwargs, kwargs))


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
    dtypes = opts.kwargs['dtypes']
    allow_nan = opts.kwargs['allow_nan']
    min_size = opts.kwargs['min_size']
    max_size = opts.kwargs['max_size']

    assert min_size <= len(result) <= max_size

    # Assert allow_nan
    if not allow_nan:
        assert not any_nan_nat_in_numpy_array(result.data)

    # Assert dtypes
    match dtypes:
        case st_ak.RecordDraws():
            drawn_kinds = {d.kind for d in dtypes.drawn}
            assert result.data.dtype.kind in drawn_kinds


def test_draw_empty() -> None:
    '''Assert that empty arrays can be drawn.'''
    find(
        st_ak.contents.numpy_array_contents(
            dtypes=None, allow_nan=False, min_size=0, max_size=10
        ),
        lambda c: len(c) == 0,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_max_size() -> None:
    '''Assert that arrays with exactly max_size elements can be drawn.'''
    max_size = 10
    find(
        st_ak.contents.numpy_array_contents(
            dtypes=None, allow_nan=False, min_size=0, max_size=max_size
        ),
        lambda c: len(c) == max_size,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_min_size() -> None:
    '''Assert that arrays with exactly min_size elements can be drawn.'''
    min_size = 5
    find(
        st_ak.contents.numpy_array_contents(
            dtypes=None, allow_nan=False, min_size=min_size, max_size=10
        ),
        lambda c: len(c) == min_size,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_nan() -> None:
    '''Assert that arrays with NaN can be drawn when allowed.'''
    float_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'f')
    find(
        st_ak.contents.numpy_array_contents(
            dtypes=float_dtypes, allow_nan=True, min_size=1, max_size=10
        ),
        lambda c: any_nan_in_numpy_array(c.data),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
