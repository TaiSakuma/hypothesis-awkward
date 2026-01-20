import math
from typing import TypedDict, cast

import numpy as np
import pytest
from hypothesis import Phase, find, given, note, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util import (
    any_nan_in_numpy_array,
    any_nan_nat_in_numpy_array,
    any_nat_in_numpy_array,
    simple_dtype_kinds_in,
)


class NumpyArraysKwargs(TypedDict, total=False):
    '''Options for `numpy_arrays()` strategy.'''

    dtype: np.dtype | st.SearchStrategy[np.dtype] | None
    allow_structured: bool
    allow_nan: bool
    max_size: int


def numpy_arrays_kwargs() -> st.SearchStrategy[NumpyArraysKwargs]:
    '''Strategy for options for `numpy_arrays()` strategy.'''
    return st.fixed_dictionaries(
        {},
        optional={
            'dtype': st.one_of(
                st.none(),
                st.just(st_ak.supported_dtypes()),
                st_ak.supported_dtypes(),
            ),
            'allow_structured': st.booleans(),
            'allow_nan': st.booleans(),
            'max_size': st.integers(min_value=0, max_value=100),
        },
    ).map(lambda d: cast(NumpyArraysKwargs, d))


@settings(max_examples=200)
@given(data=st.data())
def test_numpy_arrays(data: st.DataObject) -> None:
    # Draw options
    kwargs = data.draw(numpy_arrays_kwargs(), label='kwargs')

    # Call the test subject
    n = data.draw(st_ak.numpy_arrays(**kwargs), label='n')

    # Assert the options were effective
    dtype = kwargs.get('dtype', None)
    allow_structured = kwargs.get('allow_structured', True)
    allow_nan = kwargs.get('allow_nan', False)
    max_size = kwargs.get('max_size', 10)

    if dtype is not None and not isinstance(dtype, st.SearchStrategy):
        kinds = simple_dtype_kinds_in(n.dtype)
        assert len(kinds) == 1
        assert dtype.kind in kinds

    size = math.prod(n.shape)
    assert size <= max_size

    structured = n.dtype.names is not None
    has_nan = any_nan_nat_in_numpy_array(n)

    if not allow_structured:
        assert not structured

    if not allow_nan:
        assert not has_nan

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
        st_ak.numpy_arrays(allow_nan=True),
        lambda a: any_nan_in_numpy_array(a),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_nat() -> None:
    '''Assert that arrays with NaT can be drawn when allowed.'''
    find(
        st_ak.numpy_arrays(allow_nan=True),
        lambda a: any_nat_in_numpy_array(a),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
