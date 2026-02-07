import numpy as np
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward.strategies.numpy import numpy_arrays

MAX_REGULAR_SIZE = 5


@st.composite
def arrays(
    draw: st.DrawFn,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,
    allow_regular: bool = True,
    max_size: int = 10,
    max_depth: int = 3,
) -> ak.Array:
    '''Strategy for Awkward Arrays built from direct Content constructors.

    Parameters
    ----------
    dtypes
        A strategy for NumPy dtypes used in leaf ``NumpyArray`` nodes.
        If ``None``, uses ``supported_dtypes()``.
    allow_nan
        Generate potentially ``NaN``/``NaT`` values for relevant dtypes
        if ``True``.
    allow_regular
        Allow wrapping the leaf ``NumpyArray`` in one or more
        ``RegularArray`` layers if ``True``.
    max_size
        Maximum total number of leaf scalars in the generated array
        (i.e., the sum of ``arr.size`` across all leaf ``NumpyArray``
        nodes).
    max_depth
        Maximum number of nested ``RegularArray`` layers wrapping the
        leaf ``NumpyArray``.  Only effective when *allow_regular* is
        ``True``.

    Examples
    --------
    >>> arrays().example()
    <Array ... type='...'>

    '''
    effective_max_depth = max_depth if allow_regular else 0
    base = _numpy_leaf(dtypes, allow_nan, max_size)
    if effective_max_depth == 0:
        layout = draw(base)
    else:
        max_leaves = 2 ** (effective_max_depth - 1)
        layout = draw(st.recursive(base, _wrap_regular, max_leaves=max_leaves))
    return ak.Array(layout)


def _numpy_leaf(
    dtypes: st.SearchStrategy[np.dtype] | None,
    allow_nan: bool,
    max_size: int,
) -> st.SearchStrategy[ak.contents.NumpyArray]:
    '''Base strategy: leaf NumpyArray Content.'''
    return numpy_arrays(
        dtype=dtypes,
        allow_structured=False,
        allow_nan=allow_nan,
        max_dims=1,
        max_size=max_size,
    ).map(ak.contents.NumpyArray)


@st.composite
def _wrap_regular(
    draw: st.DrawFn,
    children: st.SearchStrategy[ak.contents.Content],
) -> ak.contents.Content:
    '''Extend strategy: wrap child Content in a RegularArray.'''
    child = draw(children)
    child_len = len(child)
    if child_len == 0:
        size = draw(st.integers(min_value=0, max_value=MAX_REGULAR_SIZE))
        if size == 0:
            zeros_length = draw(
                st.integers(min_value=0, max_value=MAX_REGULAR_SIZE)
            )
            return ak.contents.RegularArray(
                child, size=0, zeros_length=zeros_length
            )
        return ak.contents.RegularArray(child, size=size)
    divisors = [
        d
        for d in range(1, min(child_len + 1, MAX_REGULAR_SIZE + 1))
        if child_len % d == 0
    ]
    size = draw(st.sampled_from(divisors))
    return ak.contents.RegularArray(child, size=size)
