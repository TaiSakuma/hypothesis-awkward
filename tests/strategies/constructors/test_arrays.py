from typing import TypedDict, cast

import numpy as np
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
DEFAULT_MAX_DEPTH = 3


class ArraysKwargs(TypedDict, total=False):
    '''Options for `arrays()` strategy.'''

    dtypes: st.SearchStrategy[np.dtype] | None
    allow_nan: bool
    allow_regular: bool
    max_size: int
    max_depth: int


def arrays_kwargs() -> st.SearchStrategy[st_ak.Opts[ArraysKwargs]]:
    '''Strategy for options for `arrays()` strategy.'''
    return (
        st.fixed_dictionaries(
            {},
            optional={
                'dtypes': st.one_of(
                    st.none(),
                    st.just(st_ak.RecordDraws(st_ak.supported_dtypes())),
                ),
                'allow_nan': st.booleans(),
                'allow_regular': st.booleans(),
                'max_size': st.integers(min_value=0, max_value=50),
                'max_depth': st.integers(min_value=0, max_value=3),
            },
        )
        .map(lambda d: cast(ArraysKwargs, d))
        .map(st_ak.Opts)
    )


@settings(max_examples=200)
@given(data=st.data())
def test_arrays(data: st.DataObject) -> None:
    '''Test that `arrays()` respects all its options.'''
    # Draw options
    opts = data.draw(arrays_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    a = data.draw(st_ak.constructors.arrays(**opts.kwargs), label='a')

    # Assert the result is always an ak.Array
    assert isinstance(a, ak.Array)

    # Assert the options were effective
    dtypes = opts.kwargs.get('dtypes', None)
    allow_nan = opts.kwargs.get('allow_nan', False)
    allow_regular = opts.kwargs.get('allow_regular', True)
    max_size = opts.kwargs.get('max_size', DEFAULT_MAX_SIZE)
    max_depth = opts.kwargs.get('max_depth', DEFAULT_MAX_DEPTH)

    # When RegularArray is disallowed or depth is zero, layout must be NumpyArray
    if not allow_regular or max_depth == 0:
        assert isinstance(a.layout, ak.contents.NumpyArray)

    # Dtype check via leaf arrays (works for both flat and nested layouts)
    match dtypes:
        case None:
            pass
        case st_ak.RecordDraws():
            drawn_dtype_names = {d.name for d in dtypes.drawn}
            leaf_dtype_names = {d.name for d in _leaf_dtypes(a)}
            assert leaf_dtype_names <= drawn_dtype_names

    if not allow_nan:
        assert not any_nan_nat_in_awkward_array(a)

    assert _total_scalars(a) <= max_size

    assert _regular_depth(a) <= max_depth


def test_draw_empty() -> None:
    '''Assert that empty arrays can be drawn by default.'''
    find(
        st_ak.constructors.arrays(),
        lambda a: len(a) == 0,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_max_size() -> None:
    '''Assert that arrays with max_size scalars can be drawn.'''
    max_size = 20
    find(
        st_ak.constructors.arrays(max_size=max_size),
        lambda a: _total_scalars(a) == max_size,
        settings=settings(phases=[Phase.generate], max_examples=5000),
    )


def test_draw_nan() -> None:
    '''Assert that arrays with NaN can be drawn when allowed.'''
    float_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'f')
    find(
        st_ak.constructors.arrays(dtypes=float_dtypes, allow_nan=True),
        any_nan_in_awkward_array,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_integer_dtype() -> None:
    '''Assert that integer dtype arrays can be drawn.'''
    int_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'i')
    find(
        st_ak.constructors.arrays(dtypes=int_dtypes),
        lambda a: a.layout.dtype.kind == 'i',
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_max_depth() -> None:
    '''Assert that arrays at exactly max_depth levels of nesting can be drawn.'''
    max_depth = 4
    find(
        st_ak.constructors.arrays(max_depth=max_depth),
        lambda a: _regular_depth(a) == max_depth,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_regular_size_zero() -> None:
    '''Assert that RegularArray with size=0 can be drawn.'''

    def _has_regular_size_zero(a: ak.Array) -> bool:
        stack: list[ak.contents.Content] = [a.layout]
        while stack:
            node = stack.pop()
            if isinstance(node, ak.contents.RegularArray) and node.size == 0:
                return True
            if hasattr(node, 'content'):
                stack.append(node.content)
        return False

    find(
        st_ak.constructors.arrays(),
        _has_regular_size_zero,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def _total_scalars(a: ak.Array) -> int:
    '''Total number of scalar values across all leaf NumPy arrays.'''
    return sum(arr.size for arr in iter_numpy_arrays(a))


def _leaf_dtypes(a: ak.Array) -> set[np.dtype]:
    '''Dtypes of leaf NumPy arrays contained in `a`.'''
    return {arr.dtype for arr in iter_numpy_arrays(a)}


def _regular_depth(a: ak.Array) -> int:
    '''Count the depth of nested RegularArrays from the top-level layout.'''
    depth = 0
    node: ak.contents.Content = a.layout
    while isinstance(node, ak.contents.RegularArray):
        depth += 1
        node = node.content
    return depth
