from __future__ import annotations

from typing import Any, TypedDict, cast

import numpy as np
import pytest
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util import (
    any_nan_in_awkward_array,
    any_nan_nat_in_awkward_array,
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
    allow_regular: bool
    allow_list_offset: bool
    allow_list: bool
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
                'allow_regular': st.booleans(),
                'allow_list_offset': st.booleans(),
                'allow_list': st.booleans(),
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
    if not allow_numpy and not allow_empty:
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
    max_depth = opts.kwargs.get('max_depth', DEFAULT_MAX_DEPTH)

    # Flat leaf when all structural types disabled
    if not allow_regular and not allow_list_offset and not allow_list:
        assert isinstance(c, (ak.contents.NumpyArray, ak.contents.EmptyArray))

    # Per-type gating
    if not allow_regular:
        assert not _has_regular(c)
    if not allow_list_offset:
        assert not _has_list_offset(c)
    if not allow_list:
        assert not _has_list(c)

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


def test_draw_max_size() -> None:
    '''Assert that content with max_size scalars can be drawn.'''
    max_size = 8
    find(
        st_ak.contents.contents(max_size=max_size),
        lambda c: _total_scalars(c) == max_size,
        settings=settings(phases=[Phase.generate], max_examples=10000),
    )


def test_draw_nan() -> None:
    '''Assert that content with NaN can be drawn when allowed.'''
    float_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'f')
    find(
        st_ak.contents.contents(dtypes=float_dtypes, allow_nan=True),
        any_nan_in_awkward_array,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_integer_dtype() -> None:
    '''Assert that integer dtype content can be drawn.'''
    int_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'i')
    find(
        st_ak.contents.contents(dtypes=int_dtypes),
        lambda c: isinstance(c, ak.contents.NumpyArray) and c.dtype.kind == 'i',
        settings=settings(phases=[Phase.generate]),
    )


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
    '''Count total structural wrapping layers (RegularArray and ListOffsetArray).'''
    depth = 0
    node = c
    while isinstance(
        node,
        (ak.contents.RegularArray, ak.contents.ListOffsetArray, ak.contents.ListArray),
    ):
        depth += 1
        node = node.content
    return depth


def _has_list_offset(c: ak.contents.Content) -> bool:
    '''Check if the content contains any ListOffsetArray node.'''
    stack: list[ak.contents.Content] = [c]
    while stack:
        node = stack.pop()
        if isinstance(node, ak.contents.ListOffsetArray):
            return True
        if hasattr(node, 'content'):
            stack.append(node.content)
    return False


def _has_regular(c: ak.contents.Content) -> bool:
    '''Check if the content contains any RegularArray node.'''
    stack: list[ak.contents.Content] = [c]
    while stack:
        node = stack.pop()
        if isinstance(node, ak.contents.RegularArray):
            return True
        if hasattr(node, 'content'):
            stack.append(node.content)
    return False


def _has_list(c: ak.contents.Content) -> bool:
    '''Check if the content contains any ListArray node.'''
    stack: list[ak.contents.Content] = [c]
    while stack:
        node = stack.pop()
        if isinstance(node, ak.contents.ListArray):
            return True
        if hasattr(node, 'content'):
            stack.append(node.content)
    return False
