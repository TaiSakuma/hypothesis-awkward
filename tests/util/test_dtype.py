import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util import simple_dtype_kinds_in


def _is_kind_in(k: str, d: np.dtype) -> bool:
    if (fields := d.fields) is not None:
        return any(_is_kind_in(k, f[0]) for f in fields.values())
    if (subdtype := d.subdtype) is not None:
        return _is_kind_in(k, subdtype[0])
    return d.kind == k


ALL_SIMPLE_DTYPE_KINDS = {'b', 'i', 'u', 'f', 'c', 'm', 'M', 'O', 'S', 'U'}


@given(d=st_ak.numpy_dtypes())
def test_simple_dtype_kinds_in(d: np.dtype) -> None:
    kinds = simple_dtype_kinds_in(d)
    assert 'V' not in kinds
    for k in kinds:
        assert _is_kind_in(k, d)
    for k in ALL_SIMPLE_DTYPE_KINDS - kinds:
        assert not _is_kind_in(k, d)
