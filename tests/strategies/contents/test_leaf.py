from typing import Any, TypedDict, cast

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
    allow_string: bool
    allow_bytestring: bool


@st.composite
def leaf_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[LeafContentsKwargs]:
    '''Strategy for options for `leaf_contents()` strategy.'''
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
                'allow_numpy': st.booleans(),
                'allow_empty': st.booleans(),
                'allow_string': st.booleans(),
                'allow_bytestring': st.booleans(),
            },
        )
    )

    return chain.extend(cast(LeafContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_leaf_contents(data: st.DataObject) -> None:
    '''Test that `leaf_contents()` respects all its options.'''
    opts = data.draw(leaf_contents_kwargs(), label='opts')
    opts.reset()

    allow_numpy = opts.kwargs.get('allow_numpy', True)
    allow_empty = opts.kwargs.get('allow_empty', True)
    allow_string = opts.kwargs.get('allow_string', True)
    allow_bytestring = opts.kwargs.get('allow_bytestring', True)

    # If all are False, expect ValueError
    if not any((allow_numpy, allow_empty, allow_string, allow_bytestring)):
        with pytest.raises(
            ValueError, match='at least one leaf content type must be allowed'
        ):
            st_ak.contents.leaf_contents(**opts.kwargs)
        return

    result = data.draw(st_ak.contents.leaf_contents(**opts.kwargs), label='result')

    is_numpy = isinstance(result, NumpyArray)
    is_empty = isinstance(result, EmptyArray)
    is_string = result.parameter('__array__') == 'string'
    is_bytestring = result.parameter('__array__') == 'bytestring'

    assert any((is_numpy, is_empty, is_string, is_bytestring))

    if not allow_numpy:
        assert not is_numpy
    if not allow_empty:
        assert not is_empty
    if not allow_string:
        assert not is_string
    if not allow_bytestring:
        assert not is_bytestring

    dtypes = opts.kwargs.get('dtypes', None)
    allow_nan = opts.kwargs.get('allow_nan', False)
    min_size = opts.kwargs.get('min_size', 0)
    max_size = opts.kwargs.get('max_size', DEFAULT_MAX_SIZE)

    if is_empty:
        assert len(result) == 0
    else:
        assert sc(min_size) <= len(result) <= sc(max_size)

    if is_numpy:
        assert isinstance(result, NumpyArray)
        if not allow_nan:
            assert not any_nan_nat_in_numpy_array(result.data)
        match dtypes:
            case st_ak.RecordDraws():
                drawn_kinds = {d.kind for d in dtypes.drawn}
                assert result.data.dtype.kind in drawn_kinds


def test_draw_numpy_array() -> None:
    '''Assert that NumpyArray can be drawn by default.'''
    find(
        st_ak.contents.leaf_contents(),
        lambda c: isinstance(c, NumpyArray)
        and c.parameter('__array__') not in ('string', 'bytestring'),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_empty_array() -> None:
    '''Assert that EmptyArray can be drawn by default.'''
    find(
        st_ak.contents.leaf_contents(),
        lambda c: isinstance(c, EmptyArray),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_string() -> None:
    '''Assert that string content can be drawn by default.'''
    find(
        st_ak.contents.leaf_contents(),
        lambda c: c.parameter('__array__') == 'string',
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_bytestring() -> None:
    '''Assert that bytestring content can be drawn by default.'''
    find(
        st_ak.contents.leaf_contents(),
        lambda c: c.parameter('__array__') == 'bytestring',
        settings=settings(phases=[Phase.generate]),
    )
