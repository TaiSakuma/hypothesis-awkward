import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

from hypothesis_awkward.util import any_nan_nat_in_numpy_array


def _has_nan_nat_via_iteration(n: np.ndarray) -> bool:
    '''Check for NaN/NaT by iterating over flattened array.'''
    for val in n.flat:
        if isinstance(val, (complex, np.complexfloating)):
            if math.isnan(val.real) or math.isnan(val.imag):
                return True
        elif isinstance(val, (float, np.floating)):
            if math.isnan(val):
                return True
        elif isinstance(val, (np.datetime64, np.timedelta64)):
            if np.isnat(val):
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
