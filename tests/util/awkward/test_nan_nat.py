import math

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util import (
    any_nan_in_awkward_array,
    any_nan_nat_in_awkward_array,
    any_nat_in_awkward_array,
)
from hypothesis_awkward.util.awkward import iter_leaf_contents


@given(data=st.data())
def test_any_nan_nat_in_awkward_array(data: st.DataObject) -> None:
    '''Verify result matches element-by-element iteration.'''
    allow_nan = data.draw(st.booleans())
    a = data.draw(st_ak.constructors.arrays(allow_nan=allow_nan, allow_virtual=False))
    actual = any_nan_nat_in_awkward_array(a)
    if allow_nan:
        expected = _expected_any_nan_nat(a)
    else:
        expected = False
    assert actual == expected


@given(data=st.data())
def test_any_nan_in_awkward_array(data: st.DataObject) -> None:
    '''Verify result matches element-by-element iteration.'''
    allow_nan = data.draw(st.booleans())
    a = data.draw(st_ak.constructors.arrays(allow_nan=allow_nan, allow_virtual=False))
    actual = any_nan_in_awkward_array(a)
    if allow_nan:
        expected = _expected_any_nan(a)
    else:
        expected = False
    assert actual == expected


@given(data=st.data())
def test_any_nat_in_awkward_array(data: st.DataObject) -> None:
    '''Verify result matches element-by-element iteration.'''
    allow_nan = data.draw(st.booleans())
    a = data.draw(st_ak.constructors.arrays(allow_nan=allow_nan, allow_virtual=False))
    actual = any_nat_in_awkward_array(a)
    if allow_nan:
        expected = _expected_any_nat(a)
    else:
        expected = False
    assert actual == expected


def test_draw_nan() -> None:
    '''Assert that arrays with NaN can be drawn.'''
    find(
        st_ak.constructors.arrays(allow_nan=True, allow_virtual=False),
        _expected_any_nan,
        settings=settings(max_examples=10000, phases=[Phase.generate]),
    )


def test_draw_nat() -> None:
    '''Assert that arrays with NaT can be drawn.'''
    find(
        st_ak.constructors.arrays(allow_nan=True, allow_virtual=False),
        _expected_any_nat,
        settings=settings(max_examples=10000, phases=[Phase.generate]),
    )


def _expected_any_nan_nat(a: ak.Array) -> bool:
    '''Check if array contains any NaN or NaT.'''
    return _expected_any_nan(a) or _expected_any_nat(a)


def _expected_any_nan(a: ak.Array) -> bool:
    '''Check if array contains any NaN.'''
    for content in iter_leaf_contents(a):
        if not isinstance(content, ak.contents.NumpyArray):
            continue
        arr = content.data
        match arr.dtype.kind:
            case 'c':
                if any(
                    math.isnan(val.real) or math.isnan(val.imag) for val in arr.flat
                ):
                    return True
            case 'f':
                if any(math.isnan(val) for val in arr.flat):
                    return True
    return False


def _expected_any_nat(a: ak.Array) -> bool:
    '''Check if array contains any NaT.'''
    for content in iter_leaf_contents(a):
        if not isinstance(content, ak.contents.NumpyArray):
            continue
        arr = content.data
        if arr.dtype.kind not in {'m', 'M'}:
            continue
        if any(np.isnat(val) for val in arr.flat):
            return True
    return False
