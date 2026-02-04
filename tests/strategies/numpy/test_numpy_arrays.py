import math
from typing import TypedDict, cast

import numpy as np
import pytest
from hypothesis import Phase, find, given, note, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util import (
    any_nan_in_numpy_array,
    any_nan_nat_in_numpy_array,
    any_nat_in_numpy_array,
    simple_dtype_kinds_in,
)

DEFAULT_MAX_SIZE = 10


class NumpyArraysKwargs(TypedDict, total=False):
    '''Options for `numpy_arrays()` strategy.'''

    dtype: np.dtype | st.SearchStrategy[np.dtype] | None
    allow_structured: bool
    allow_nan: bool
    allow_inner_shape: bool
    max_size: int


class NumpyArraysOpts:
    '''Drawn options for `numpy_arrays()` with resettable recorders.'''

    def __init__(self, kwargs: NumpyArraysKwargs) -> None:
        self._kwargs = kwargs

    @property
    def kwargs(self) -> NumpyArraysKwargs:
        return self._kwargs

    def reset(self) -> None:
        for v in self._kwargs.values():
            if isinstance(v, st_ak.RecordDraws):
                v.drawn.clear()


def numpy_arrays_kwargs() -> st.SearchStrategy[NumpyArraysOpts]:
    '''Strategy for options for `numpy_arrays()` strategy.'''
    return (
        st.fixed_dictionaries(
            {},
            optional={
                'dtype': st.one_of(
                    st.none(),
                    st.just(st_ak.RecordDraws(st_ak.supported_dtypes())),
                    st_ak.supported_dtypes(),
                ),
                'allow_structured': st.booleans(),
                'allow_nan': st.booleans(),
                'allow_inner_shape': st.booleans(),
                'max_size': st.integers(min_value=0, max_value=100),
            },
        )
        .map(lambda d: cast(NumpyArraysKwargs, d))
        .map(NumpyArraysOpts)
    )


@settings(max_examples=200)
@given(data=st.data())
def test_numpy_arrays(data: st.DataObject) -> None:
    # Draw options
    opts = data.draw(numpy_arrays_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    n = data.draw(st_ak.numpy_arrays(**opts.kwargs), label='n')

    # Assert the options were effective
    dtype = opts.kwargs.get('dtype', None)
    allow_structured = opts.kwargs.get('allow_structured', True)
    allow_nan = opts.kwargs.get('allow_nan', False)
    allow_inner_shape = opts.kwargs.get('allow_inner_shape', True)
    max_size = opts.kwargs.get('max_size', DEFAULT_MAX_SIZE)

    match dtype:
        case np.dtype():
            kinds = simple_dtype_kinds_in(n.dtype)
            assert len(kinds) == 1
            assert dtype.kind in kinds
        case st_ak.RecordDraws():
            drawn_kinds = {d.kind for d in dtype.drawn}
            result_kinds = simple_dtype_kinds_in(n.dtype)
            assert result_kinds <= drawn_kinds

    size = math.prod(n.shape)
    assert size <= max_size

    structured = n.dtype.names is not None
    has_nan = any_nan_nat_in_numpy_array(n)

    if not allow_structured:
        assert not structured

    if not allow_nan:
        assert not has_nan

    if not allow_inner_shape:
        assert len(n.shape) == 1

    # Assert an Awkward Array can be created.
    a = ak.from_numpy(n)
    note(f'{a=}')
    assert isinstance(a, ak.Array)

    # Test if the NumPy array and Awkward Array are converted to the same list.
    # Compare only when `NaN` isn't allowed.
    # Structured arrays are known to result in a different list sometimes.
    to_list = a.to_list()
    note(f'{to_list=}')

    if not allow_nan:
        if not structured:  # simple array
            assert to_list == n.tolist()
        else:  # structured array
            # assert to_list == n.tolist()  # NOTE: Fails sometimes
            pass

    # Test if the Awkward Array is converted back to a NumPy array with the identical
    # values. The conversion of structured arrays fails under a known condition.
    # Structured arrays may not result in identical values.

    def _is_numpy_convertible(a: ak.Array) -> bool:
        '''True if `a.to_numpy()` is expected to work without error.

        `to_numpy()` fails for structured arrays with non-1D fields
        https://github.com/scikit-hep/awkward/issues/3690


        '''
        layout = a.layout
        if isinstance(layout, ak.contents.NumpyArray):  # simple array
            return True
        assert isinstance(layout, ak.contents.RecordArray)  # structured array
        return all(len(c.shape) == 1 for c in layout.contents)

    if _is_numpy_convertible(a):
        to_numpy = a.to_numpy()
        note(f'{to_numpy=}')
        if not has_nan:
            if not structured:
                np.testing.assert_array_equal(to_numpy, n)
            else:
                # np.testing.assert_array_equal(to_numpy, n)  # NOTE: Fails sometimes
                pass
    else:
        with pytest.raises(ValueError):
            a.to_numpy()


def test_draw_structured() -> None:
    '''Assert that structured arrays can be drawn by default.'''
    find(
        st_ak.numpy_arrays(),
        lambda a: a.dtype.names is not None,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_nan() -> None:
    '''Assert that arrays with NaN can be drawn when allowed.'''
    find(
        st_ak.numpy_arrays(dtype=st_np.floating_dtypes(), allow_nan=True),
        lambda a: any_nan_in_numpy_array(a),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_nat_datetime64() -> None:
    '''Assert that datetime64 arrays with NaT can be drawn when allowed.'''
    find(
        st_ak.numpy_arrays(dtype=st_np.datetime64_dtypes(), allow_nan=True),
        lambda a: any_nat_in_numpy_array(a),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_nat_timedelta64() -> None:
    '''Assert that timedelta64 arrays with NaT can be drawn when allowed.'''
    find(
        st_ak.numpy_arrays(dtype=st_np.timedelta64_dtypes(), allow_nan=True),
        lambda a: any_nat_in_numpy_array(a),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_empty() -> None:
    '''Assert that empty arrays can be drawn by default.'''
    find(
        st_ak.numpy_arrays(),
        lambda a: math.prod(a.shape) == 0,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_max_size() -> None:
    '''Assert that arrays with max_size elements can be drawn by default.'''
    find(
        st_ak.numpy_arrays(allow_structured=False),
        lambda a: math.prod(a.shape) == DEFAULT_MAX_SIZE,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_inner_shape() -> None:
    '''Assert that multi-dimensional arrays can be drawn by default.'''
    find(
        st_ak.numpy_arrays(allow_structured=False),
        lambda a: len(a.shape) > 1,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
