from __future__ import annotations

from typing import Any, TypedDict, cast

import numpy as np
import pytest
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util import (
    any_nan_nat_in_awkward_array,
    iter_contents,
    iter_numpy_arrays,
)

DEFAULT_MAX_SIZE = 10
DEFAULT_MAX_DEPTH = 5


class ContentsKwargs(TypedDict, total=False):
    '''Options for `contents()` strategy.'''

    dtypes: st.SearchStrategy[np.dtype] | None
    max_size: int
    allow_nan: bool
    allow_numpy: bool
    allow_empty: bool
    allow_string: bool
    allow_bytestring: bool
    allow_regular: bool
    allow_list_offset: bool
    allow_list: bool
    allow_record: bool
    allow_union: bool
    max_depth: int


@st.composite
def contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[ContentsKwargs]:
    '''Strategy for options for `contents()` strategy.'''
    if chain is None:
        chain = st_ak.OptsChain({})
    st_dtypes = chain.register(st_ak.supported_dtypes())

    kwargs = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'dtypes': st.one_of(
                    st.none(),
                    st.just(st_dtypes),
                ),
                'max_size': st.integers(min_value=0, max_value=50),
                'allow_nan': st.booleans(),
                'allow_numpy': st.booleans(),
                'allow_empty': st.booleans(),
                'allow_string': st.booleans(),
                'allow_bytestring': st.booleans(),
                'allow_regular': st.booleans(),
                'allow_list_offset': st.booleans(),
                'allow_list': st.booleans(),
                'allow_record': st.booleans(),
                'allow_union': st.booleans(),
                'max_depth': st.integers(min_value=0, max_value=5),
            },
        )
    )

    return chain.extend(cast(ContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_contents(data: st.DataObject) -> None:
    '''Test that `contents()` respects all its options.'''
    # Draw options
    opts = data.draw(contents_kwargs(), label='opts')
    opts.reset()

    # Assert that disabling all leaf types raises an error
    allow_numpy = opts.kwargs.get('allow_numpy', True)
    allow_empty = opts.kwargs.get('allow_empty', True)
    allow_string = opts.kwargs.get('allow_string', True)
    allow_bytestring = opts.kwargs.get('allow_bytestring', True)
    allow_any_leaf = any((allow_numpy, allow_empty, allow_string, allow_bytestring))
    if not allow_any_leaf:
        with pytest.raises(ValueError, match='at least one leaf'):
            data.draw(st_ak.contents.contents(**opts.kwargs), label='c')
        return

    # Call the test subject
    c = data.draw(st_ak.contents.contents(**opts.kwargs), label='c')

    # Assert the result is always an ak.contents.Content
    assert isinstance(c, ak.contents.Content)

    # Assert the options were effective
    dtypes = opts.kwargs.get('dtypes', None)
    max_size = opts.kwargs.get('max_size', DEFAULT_MAX_SIZE)
    allow_nan = opts.kwargs.get('allow_nan', False)
    allow_regular = opts.kwargs.get('allow_regular', True)
    allow_list_offset = opts.kwargs.get('allow_list_offset', True)
    allow_list = opts.kwargs.get('allow_list', True)
    allow_record = opts.kwargs.get('allow_record', True)
    allow_union = opts.kwargs.get('allow_union', True)
    max_depth = opts.kwargs.get('max_depth', DEFAULT_MAX_DEPTH)

    allow_any_nesting = any(
        (allow_regular, allow_list_offset, allow_list, allow_record, allow_union)
    )
    if not allow_any_nesting:
        assert _nesting_depth(c) == 0

    # Per-type gating
    if not allow_numpy:
        assert not _has_numpy(c)
    if not allow_empty:
        assert not _has_empty(c)
    if not allow_regular:
        assert not _has_regular(c)
    if not allow_list_offset:
        assert not _has_list_offset(c)
    if not allow_list:
        assert not _has_list(c)
    if not allow_string:
        assert not _has_string(c)
    if not allow_bytestring:
        assert not _has_bytestring(c)
    if not allow_record:
        assert not _has_record(c)
    if not allow_union:
        assert not _has_union(c)

    # Dtype check via leaf arrays (works for both flat and nested layouts)
    match dtypes:
        case None:
            pass
        case st_ak.RecordDraws():
            drawn_dtype_names = {d.name for d in dtypes.drawn}
            leaf_dtype_names = {d.name for d in _leaf_dtypes(c)}
            assert leaf_dtype_names <= drawn_dtype_names

    assert _total_scalars(c) <= max_size

    if not allow_nan:
        assert not any_nan_nat_in_awkward_array(c)

    assert _nesting_depth(c) <= max_depth


def test_draw_max_depth() -> None:
    '''Assert that content at exactly max_depth can be drawn.'''
    max_depth = 8
    find(
        st_ak.contents.contents(max_size=20, max_depth=max_depth),
        lambda c: _nesting_depth(c) == max_depth,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_nested() -> None:
    '''Assert that nested content (depth >= 2) can be drawn.'''
    find(
        st_ak.contents.contents(max_size=20),
        lambda c: _nesting_depth(c) >= 2,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def _total_scalars(c: ak.contents.Content) -> int:
    '''Total number of scalar values across all leaf NumPy arrays.'''
    return sum(arr.size for arr in iter_numpy_arrays(c))


def _leaf_dtypes(c: ak.contents.Content) -> set[np.dtype]:
    '''Dtypes of leaf NumPy arrays contained in `c`.'''
    return {arr.dtype for arr in iter_numpy_arrays(c)}


def _nesting_depth(c: ak.contents.Content) -> int:
    '''Maximum structural nesting depth (deepest path through the tree).'''
    _nesting_types = (
        ak.contents.RegularArray,
        ak.contents.ListOffsetArray,
        ak.contents.ListArray,
    )
    if isinstance(c, ak.contents.UnionArray):
        return 1 + max(_nesting_depth(child) for child in c.contents)
    if isinstance(c, ak.contents.RecordArray):
        if not c.contents:
            return 1
        return 1 + max(_nesting_depth(child) for child in c.contents)
    if isinstance(c, _nesting_types):
        if c.parameter('__array__') in ('string', 'bytestring'):
            return 0
        return 1 + _nesting_depth(c.content)
    return 0


def _has_numpy(c: ak.contents.Content) -> bool:
    '''Check if the content contains any NumpyArray node.'''
    return any(isinstance(n, ak.contents.NumpyArray) for n in iter_contents(c))


def _has_empty(c: ak.contents.Content) -> bool:
    '''Check if the content contains any EmptyArray node.'''
    return any(isinstance(n, ak.contents.EmptyArray) for n in iter_contents(c))


def _has_list_offset(c: ak.contents.Content) -> bool:
    '''Check if the content contains any structural ListOffsetArray node.'''
    return any(
        isinstance(n, ak.contents.ListOffsetArray)
        and n.parameter('__array__') not in ('string', 'bytestring')
        for n in iter_contents(c)
    )


def _has_regular(c: ak.contents.Content) -> bool:
    '''Check if the content contains any RegularArray node.'''
    return any(isinstance(n, ak.contents.RegularArray) for n in iter_contents(c))


def _has_list(c: ak.contents.Content) -> bool:
    '''Check if the content contains any ListArray node.'''
    return any(isinstance(n, ak.contents.ListArray) for n in iter_contents(c))


def _has_string(c: ak.contents.Content) -> bool:
    '''Check if the content contains any string node.'''
    return any(n.parameter('__array__') == 'string' for n in iter_contents(c))


def _has_bytestring(c: ak.contents.Content) -> bool:
    '''Check if the content contains any bytestring node.'''
    return any(n.parameter('__array__') == 'bytestring' for n in iter_contents(c))


def _has_record(c: ak.contents.Content) -> bool:
    '''Check if the content contains any RecordArray node.'''
    return any(isinstance(n, ak.contents.RecordArray) for n in iter_contents(c))


def _has_union(c: ak.contents.Content) -> bool:
    '''Check if the content contains any UnionArray node.'''
    return any(isinstance(n, ak.contents.UnionArray) for n in iter_contents(c))
