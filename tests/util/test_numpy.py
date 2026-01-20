import math
from typing import TypeAlias

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np
from numpy.typing import NDArray

from hypothesis_awkward.util import any_nan_nat_in_numpy_array

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
        expected = _has_nan_nat_via_iteration(n)
    else:
        expected = False
    assert actual == expected


def test_draw_nan() -> None:
    '''Assert that arrays with NaN can be drawn by default.'''
    find(
        st_np.arrays(dtype=st_np.nested_dtypes(), shape=st_np.array_shapes()),
        lambda a: _has_nan_via_iteration(a),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_nat() -> None:
    '''Assert that arrays with NaT can be drawn by default.'''
    find(
        st_np.arrays(dtype=st_np.nested_dtypes(), shape=st_np.array_shapes()),
        lambda a: _has_nat_via_iteration(a),
        settings=settings(phases=[Phase.generate]),
    )


def _has_nan_nat_via_iteration(n: np.ndarray) -> bool:
    '''Check if array contains any NaN or NaT.'''
    return _has_nan_via_iteration(n) or _has_nat_via_iteration(n)


def _has_nan_via_iteration(n: np.ndarray) -> bool:
    '''Check for NaN by iterating over flattened array.'''
    for val in n.flat:
        assert isinstance(val, (float, complex, np.generic, np.ndarray))
        if _is_nan(val):
            return True
    return False


def _has_nat_via_iteration(n: np.ndarray) -> bool:
    '''Check for NaT by iterating over flattened array.'''
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
