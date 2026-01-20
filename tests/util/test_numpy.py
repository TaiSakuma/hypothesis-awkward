import math

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

from hypothesis_awkward.util import any_nan_nat_in_numpy_array


def _is_nan_nat(val: object) -> bool:
    '''Check if a single value is NaN or NaT.'''
    if isinstance(val, (complex, np.complexfloating)):
        return math.isnan(val.real) or math.isnan(val.imag)
    elif isinstance(val, (float, np.floating)):
        return math.isnan(val)
    elif isinstance(val, (np.datetime64, np.timedelta64)):
        return np.isnat(val)
    elif isinstance(val, np.void):
        for field in val.dtype.names:
            if _is_nan_nat(val[field]):
                return True
    return False


def _has_nan_nat_via_iteration(n: np.ndarray) -> bool:
    '''Check for NaN/NaT by iterating over flattened array.'''
    for val in n.flat:
        if _is_nan_nat(val):
            return True
    return False


@given(data=st.data())
def test_any_nan_nat_in_numpy_array(data: st.DataObject) -> None:
    '''Result should match element-by-element iteration.'''
    allow_nan = data.draw(st.booleans())
    n = data.draw(
        st_np.arrays(
            dtype=st_np.scalar_dtypes(),
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
        st_np.arrays(dtype=st_np.array_dtypes(), shape=st_np.array_shapes()),
        lambda a: _has_nan_nat_via_iteration(a),
        settings=settings(phases=[Phase.generate]),
    )
