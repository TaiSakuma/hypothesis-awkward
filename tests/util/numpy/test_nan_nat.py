import math
from typing import TypeAlias

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np
from numpy.typing import NDArray

from hypothesis_awkward.util import (
    any_nan_in_numpy_array,
    any_nan_nat_in_numpy_array,
    any_nat_in_numpy_array,
)

_ArrayElement: TypeAlias = float | complex | np.generic | NDArray[np.generic]


@given(data=st.data())
def test_any_nan_nat_in_numpy_array(data: st.DataObject) -> None:
    '''Verify result matches element-by-element iteration.'''
    allow_nan = data.draw(st.booleans())
    n = data.draw(
        st_np.arrays(
            dtype=st_np.nested_dtypes(),
            shape=st_np.array_shapes(),
            elements={'allow_nan': allow_nan},
        )
    )
    actual = any_nan_nat_in_numpy_array(n)
    if allow_nan:
        expected = _expected_any_nan_nat(n)
    else:
        expected = False
    assert actual == expected


@given(data=st.data())
def test_any_nan_in_numpy_array(data: st.DataObject) -> None:
    '''Verify result matches element-by-element iteration.'''
    allow_nan = data.draw(st.booleans())
    n = data.draw(
        st_np.arrays(
            dtype=st_np.nested_dtypes(),
            shape=st_np.array_shapes(),
            elements={'allow_nan': allow_nan},
        )
    )
    actual = any_nan_in_numpy_array(n)
    expected = _expected_any_nan(n) if allow_nan else False
    assert actual == expected


@given(data=st.data())
def test_any_nat_in_numpy_array(data: st.DataObject) -> None:
    '''Verify result matches element-by-element iteration.'''
    allow_nan = data.draw(st.booleans())
    n = data.draw(
        st_np.arrays(
            dtype=st_np.nested_dtypes(),
            shape=st_np.array_shapes(),
            elements={'allow_nan': allow_nan},
        )
    )
    actual = any_nat_in_numpy_array(n)
    expected = _expected_any_nat(n) if allow_nan else False
    assert actual == expected


def test_draw_nan() -> None:
    '''Assert that arrays with NaN can be drawn by default.'''
    find(
        st_np.arrays(dtype=st_np.nested_dtypes(), shape=st_np.array_shapes()),
        _expected_any_nan,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_nat() -> None:
    '''Assert that arrays with NaT can be drawn by default.'''
    find(
        st_np.arrays(dtype=st_np.nested_dtypes(), shape=st_np.array_shapes()),
        _expected_any_nat,
        settings=settings(phases=[Phase.generate]),
    )


def _expected_any_nan_nat(n: np.ndarray) -> bool:
    '''Check if array contains any NaN or NaT.'''
    return _expected_any_nan(n) or _expected_any_nat(n)


def _expected_any_nan(n: np.ndarray) -> bool:
    '''Check if array contains any NaN.'''
    for val in n.flat:
        assert isinstance(val, (float, complex, np.generic, np.ndarray))
        if _is_nan(val):
            return True
    return False


def _expected_any_nat(n: np.ndarray) -> bool:
    '''Check if array contains any NaT.'''
    for val in n.flat:
        assert isinstance(val, (float, complex, np.generic, np.ndarray))
        if _is_nat(val):
            return True
    return False


def _is_nan(val: _ArrayElement) -> bool:
    '''Check if val contains any NaN.'''
    stack: list[_ArrayElement] = [val]
    while stack:
        v = stack.pop()
        if isinstance(v, (complex, np.complexfloating)):
            if math.isnan(v.real) or math.isnan(v.imag):
                return True
        elif isinstance(v, (float, np.floating)):
            if math.isnan(v):
                return True
        elif isinstance(v, np.ndarray):
            stack.extend(v.flat)
        elif isinstance(v, np.void):
            if v.dtype.names is not None:
                stack.extend(v[field] for field in v.dtype.names)
    return False


def _is_nat(val: _ArrayElement) -> bool:
    '''Check if val contains any NaT.'''
    stack: list[_ArrayElement] = [val]
    while stack:
        v = stack.pop()
        if isinstance(v, (np.datetime64, np.timedelta64)):
            if np.isnat(v):
                return True
        elif isinstance(v, np.ndarray):
            stack.extend(v.flat)
        elif isinstance(v, np.void):
            if v.dtype.names is not None:
                stack.extend(v[field] for field in v.dtype.names)
    return False
