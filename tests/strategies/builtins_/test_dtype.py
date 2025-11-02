from typing import Any

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak


@given(data=st.data())
def test_items_from_dtype(data: st.DataObject) -> None:
    dtype = data.draw(st_ak.builtin_safe_dtypes(), label='dtype')
    item = data.draw(st_ak.items_from_dtype(dtype), label='item')

    assert type(item).__module__ in ('builtins', 'datetime')

    def _to_dtype_to_item(dtype: np.dtype, item: Any) -> Any:
        if dtype.kind == 'M':  # datetime64
            assert not isinstance(item, int)
            # datetime64 requires the unit.
            unit, _ = np.datetime_data(dtype)
            n = dtype.type(item, unit)
        else:
            n = dtype.type(item)
        return n.item()

    assert _to_dtype_to_item(dtype, item) == item
