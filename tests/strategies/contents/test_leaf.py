from typing import TypedDict, cast

import numpy as np
import pytest
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import EmptyArray, NumpyArray
from hypothesis_awkward.util import any_nan_nat_in_numpy_array
from hypothesis_awkward.util.safe import safe_compare as sc

DEFAULT_MAX_SIZE = 10


class LeafContentsKwargs(TypedDict, total=False):
    '''Options for `leaf_contents()` strategy.'''

    dtypes: st.SearchStrategy[np.dtype] | None
    allow_nan: bool
    min_size: int
    max_size: int
    allow_numpy: bool
    allow_empty: bool


@st.composite
def leaf_contents_kwargs(draw: st.DrawFn) -> st_ak.Opts[LeafContentsKwargs]:
    '''Strategy for options for `leaf_contents()` strategy.'''

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
                    st.just(st_ak.RecordDraws(st_ak.supported_dtypes())),
                ),
                'allow_nan': st.booleans(),
                'allow_numpy': st.booleans(),
                'allow_empty': st.booleans(),
            },
        )
    )

    return st_ak.Opts(cast(LeafContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_leaf_contents(data: st.DataObject) -> None:
    '''Test that `leaf_contents()` respects all its options.'''
    opts = data.draw(leaf_contents_kwargs(), label='opts')
    opts.reset()

    allow_numpy = opts.kwargs.get('allow_numpy', True)
    allow_empty = opts.kwargs.get('allow_empty', True)

    # If both are False, expect ValueError
    if not allow_numpy and not allow_empty:
        with pytest.raises(
            ValueError, match='at least one leaf content type must be allowed'
        ):
            st_ak.contents.leaf_contents(**opts.kwargs)
        return

    result = data.draw(st_ak.contents.leaf_contents(**opts.kwargs), label='result')

    # Assert the result is always NumpyArray or EmptyArray
    assert isinstance(result, (NumpyArray, EmptyArray))

    # Assert allow_numpy / allow_empty flags
    if not allow_numpy:
        assert isinstance(result, EmptyArray)
    if not allow_empty:
        assert isinstance(result, NumpyArray)

    dtypes = opts.kwargs.get('dtypes', None)
    allow_nan = opts.kwargs.get('allow_nan', False)
    min_size = opts.kwargs.get('min_size', 0)
    max_size = opts.kwargs.get('max_size', DEFAULT_MAX_SIZE)

    if isinstance(result, NumpyArray):
        # Assert size bounds
        assert sc(min_size) <= len(result) <= sc(max_size)

        # Assert allow_nan
        if not allow_nan:
            assert not any_nan_nat_in_numpy_array(result.data)

        # Assert dtypes
        match dtypes:
            case st_ak.RecordDraws():
                drawn_kinds = {d.kind for d in dtypes.drawn}
                assert result.data.dtype.kind in drawn_kinds

    if isinstance(result, EmptyArray):
        assert len(result) == 0


def test_draw_numpy_array() -> None:
    '''Assert that NumpyArray can be drawn by default.'''
    find(
        st_ak.contents.leaf_contents(),
        lambda c: isinstance(c, NumpyArray),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_empty_array() -> None:
    '''Assert that EmptyArray can be drawn by default.'''
    find(
        st_ak.contents.leaf_contents(),
        lambda c: isinstance(c, EmptyArray),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_empty_array_only() -> None:
    '''Assert that with allow_numpy=False, only EmptyArray is drawn.'''
    find(
        st_ak.contents.leaf_contents(allow_numpy=False),
        lambda c: isinstance(c, EmptyArray),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_numpy_array_only() -> None:
    '''Assert that with allow_empty=False, only NumpyArray is drawn.'''
    find(
        st_ak.contents.leaf_contents(allow_empty=False),
        lambda c: isinstance(c, NumpyArray),
        settings=settings(phases=[Phase.generate]),
    )


@settings(max_examples=200)
@given(data=st.data())
def test_no_empty_when_min_size_positive(data: st.DataObject) -> None:
    '''Assert that EmptyArray is never drawn when min_size > 0.'''
    result = data.draw(st_ak.contents.leaf_contents(min_size=1), label='result')
    assert isinstance(result, NumpyArray)
