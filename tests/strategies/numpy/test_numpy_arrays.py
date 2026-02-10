import math
from typing import TypedDict, cast

import numpy as np
import pytest
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util import (
    any_nan_in_numpy_array,
    any_nan_nat_in_numpy_array,
    any_nat_in_numpy_array,
    n_scalars_in,
    simple_dtype_kinds_in,
)
from hypothesis_awkward.util.safe import safe_compare as sc

DEFAULT_MAX_SIZE = 10


class NumpyArraysKwargs(TypedDict, total=False):
    '''Options for `numpy_arrays()` strategy.'''

    dtype: np.dtype | st.SearchStrategy[np.dtype] | None
    allow_structured: bool
    allow_nan: bool
    min_dims: int
    max_dims: int
    min_size: int
    max_size: int


@st.composite
def numpy_arrays_kwargs(draw: st.DrawFn) -> st_ak.Opts[NumpyArraysKwargs]:
    '''Strategy for options for `numpy_arrays()` strategy.'''

    min_dims, max_dims = draw(st_ak.ranges(min_start=1, max_end=5))
    min_size, max_size = draw(
        st_ak.ranges(min_start=0, max_end=100, max_start=DEFAULT_MAX_SIZE)
    )

    drawn = (
        ('min_dims', min_dims),
        ('max_dims', max_dims),
        ('min_size', min_size),
        ('max_size', max_size),
    )

    kwargs = draw(
        st.fixed_dictionaries(
            {k: st.just(v) for k, v in drawn if v is not None},
            optional={
                'dtype': st.one_of(
                    st.none(),
                    st.just(st_ak.RecordDraws(st_ak.supported_dtypes())),
                    st_ak.supported_dtypes(),
                ),
                'allow_structured': st.booleans(),
                'allow_nan': st.booleans(),
            },
        )
    )

    return st_ak.Opts(cast(NumpyArraysKwargs, kwargs))


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
    min_dims = opts.kwargs.get('min_dims', 1)
    max_dims = opts.kwargs.get('max_dims', None)
    min_size = opts.kwargs.get('min_size', 0)
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

    n_scalars = math.prod(n.shape) * n_scalars_in(n.dtype)
    assert min_size <= n_scalars <= max_size

    structured = n.dtype.names is not None
    has_nan = any_nan_nat_in_numpy_array(n)

    if not allow_structured:
        assert not structured

    if not allow_nan:
        assert not has_nan

    assert min_dims <= len(n.shape) <= sc(max_dims)

    # Assert an Awkward Array can be created.
    a = ak.from_numpy(n)
    assert isinstance(a, ak.Array)

    # Test if the NumPy array and Awkward Array are converted to the same list.
    # Compare only when `NaN` isn't allowed.
    # Structured arrays are known to result in a different list sometimes.
    to_list = a.to_list()

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
    '''Assert that arrays with exactly max_size scalars can be drawn.'''
    find(
        st_ak.numpy_arrays(allow_structured=False),
        lambda a: math.prod(a.shape) == DEFAULT_MAX_SIZE,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_max_size_structured() -> None:
    '''Assert that max_size counts scalars for structured dtypes.'''
    find(
        st_ak.numpy_arrays(),
        lambda a: (
            math.prod(a.shape) * n_scalars_in(a.dtype) == DEFAULT_MAX_SIZE
            and a.dtype.names is not None
        ),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_min_size() -> None:
    '''Assert that arrays with exactly min_size scalars can be drawn.'''
    min_size = 5
    find(
        st_ak.numpy_arrays(allow_structured=False, min_size=min_size),
        lambda a: math.prod(a.shape) == min_size,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_min_size_structured() -> None:
    '''Assert that min_size counts scalars for structured dtypes.'''
    min_size = 5
    find(
        st_ak.numpy_arrays(min_size=min_size),
        lambda a: (
            math.prod(a.shape) * n_scalars_in(a.dtype) == min_size
            and a.dtype.names is not None
        ),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_one_dim() -> None:
    '''Assert that 1-D arrays can be drawn by default.'''
    find(
        st_ak.numpy_arrays(allow_structured=False),
        lambda a: len(a.shape) == 1,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_min_dims() -> None:
    '''Assert that arrays with at least min_dims dimensions can be drawn.'''
    find(
        st_ak.numpy_arrays(allow_structured=False, min_dims=2),
        lambda a: len(a.shape) == 2,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_max_dims() -> None:
    '''Assert that arrays with max_dims dimensions can be drawn.'''
    find(
        st_ak.numpy_arrays(allow_structured=False, max_dims=3),
        lambda a: len(a.shape) == 3,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
