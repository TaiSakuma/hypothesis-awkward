import math

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

from hypothesis_awkward.util import any_nan_nat_in_numpy_array


def _is_nan(val: object) -> bool:
    '''Check if a single value is NaN.'''
    if isinstance(val, (complex, np.complexfloating)):
        return math.isnan(val.real) or math.isnan(val.imag)
    elif isinstance(val, (float, np.floating)):
        return math.isnan(val)
    elif isinstance(val, np.ndarray):
        return any(_is_nan(item) for item in val.flat)
    elif isinstance(val, np.void):
        if val.dtype.names is not None:
            return any(_is_nan(val[field]) for field in val.dtype.names)
    return False


def _is_nat(val: object) -> bool:
    '''Check if a single value is NaT.'''
    if isinstance(val, (np.datetime64, np.timedelta64)):
        return np.isnat(val)
    elif isinstance(val, np.ndarray):
        return any(_is_nat(item) for item in val.flat)
    elif isinstance(val, np.void):
        if val.dtype.names is not None:
            return any(_is_nat(val[field]) for field in val.dtype.names)
    return False


def _has_nan_nat_via_iteration(n: np.ndarray) -> bool:
    '''Check for NaN/NaT by iterating over flattened array.'''
    return _has_nan_via_iteration(n) or _has_nat_via_iteration(n)


def _has_nan_via_iteration(n: np.ndarray) -> bool:
    '''Check for NaN by iterating over flattened array.'''
    for val in n.flat:
        if _is_nan(val):
            return True
    return False


def _has_nat_via_iteration(n: np.ndarray) -> bool:
    '''Check for NaT by iterating over flattened array.'''
    for val in n.flat:
        if _is_nat(val):
            return True
    return False


@given(data=st.data())
def test_any_nan_nat_in_numpy_array(data: st.DataObject) -> None:
    '''Result should match element-by-element iteration.'''
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
