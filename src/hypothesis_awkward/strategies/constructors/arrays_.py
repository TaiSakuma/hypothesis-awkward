import numpy as np
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward.strategies.numpy import numpy_arrays


@st.composite
def arrays(
    draw: st.DrawFn,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,
    max_length: int = 5,
) -> ak.Array:
    '''Strategy for Awkward Arrays built from direct Content constructors.

    The initial version generates only flat ``NumpyArray``-backed arrays.

    Parameters
    ----------
    dtypes
        A strategy for NumPy dtypes used in leaf ``NumpyArray`` nodes.
        If ``None``, uses ``supported_dtypes()``.
    allow_nan
        Generate potentially ``NaN``/``NaT`` values for relevant dtypes
        if ``True``.
    max_length
        Maximum number of elements in the outermost array dimension
        (i.e., ``len(result)``).

    Examples
    --------
    >>> arrays().example()
    <Array ... type='...'>

    '''
    data = draw(
        numpy_arrays(
            dtype=dtypes,
            allow_structured=False,
            allow_nan=allow_nan,
            max_dims=1,
            max_size=max_length,
        )
    )

    layout = ak.contents.NumpyArray(data)
    return ak.Array(layout)
