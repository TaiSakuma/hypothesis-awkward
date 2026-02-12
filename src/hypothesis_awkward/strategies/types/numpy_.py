import numpy as np
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward.util import SUPPORTED_DTYPES


def numpy_types(
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_datetime: bool = True,
) -> st.SearchStrategy[ak.types.NumpyType]:
    '''Strategy for NumpyType (primitive/leaf types).

    Parameters
    ----------
    dtypes
        Strategy for NumPy dtypes. If None, uses `supported_dtypes()`.
    allow_datetime
        Include datetime64/timedelta64 when `dtypes` is None.

    Examples
    --------
    >>> import hypothesis_awkward.strategies as st_ak
    >>> t = st_ak.numpy_types().example()
    >>> isinstance(t, ak.types.NumpyType)
    True
    '''
    if dtypes is None:
        if allow_datetime:
            dtypes = st.sampled_from(SUPPORTED_DTYPES)
        else:
            dtypes = st.sampled_from(
                tuple(d for d in SUPPORTED_DTYPES if d.kind not in ('M', 'm'))
            )

    return dtypes.map(lambda d: ak.types.NumpyType(d.name))
