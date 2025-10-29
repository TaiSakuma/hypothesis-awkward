import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

import awkward as ak
from awkward.types.numpytype import _primitive_to_dtype_dict, primitive_to_dtype


def _supported_dtype_names() -> tuple[str, ...]:
    '''Return names of NumPy scalar dtypes supported by Awkward Array.

    I.e., ('int32', 'float64', 'datetime64[ns]', ...)
    '''
    DATETIME_UNITS = tuple('Y M W D h m s ms us ns ps fs as'.split())

    # ('bool', 'int8', ...)
    base = tuple(
        n
        for n, d in _primitive_to_dtype_dict.items()
        if d.kind not in ('M', 'm')  # Exclude datetime/timedelta as they need units
    )

    # ('datetime64[Y]', 'datetime64[M]', ...)
    dt = tuple(f'datetime64[{unit}]' for unit in DATETIME_UNITS)

    # ('timedelta64[Y]', 'timedelta64[M]', ...)
    td = tuple(f'timedelta64[{unit}]' for unit in DATETIME_UNITS)

    return base + dt + td


# Names of NumPy dtypes supported by Awkward Array
# ('bool', 'int8', 'float16', 'datetime64[ns]', ...)
SUPPORTED_DTYPE_NAMES = _supported_dtype_names()

# NumPy dtypes supported by Awkward Array
# (dtype('bool'), dtype('int8'), dtype('float16'), dtype('datetime64[ns]'), ...)
SUPPORTED_DTYPES = tuple[np.dtype, ...](
    primitive_to_dtype(name) for name in SUPPORTED_DTYPE_NAMES
)


def supported_dtype_names() -> st.SearchStrategy[str]:
    '''Strategy for names of NumPy dtypes supported by Awkward Array.

    Examples
    --------
    >>> supported_dtype_names().example()
    '...'
    '''
    return st.sampled_from(SUPPORTED_DTYPE_NAMES)


def supported_dtypes() -> st.SearchStrategy[np.dtype]:
    '''Strategy for NumPy dtypes supported by Awkward Array.

    Examples
    --------
    >>> supported_dtypes().example()
    dtype(...)
    '''
    return st.sampled_from(SUPPORTED_DTYPES)


def numpy_dtypes(
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None = None,
    allow_array: bool = True,
) -> st.SearchStrategy[np.dtype]:
    '''Strategy for dtypes (simple or array) supported by Awkward Array.

    Examples of simple dtypes are dtype('int32'), dtype('float64')

    Array dtypes include ([('f0', 'i4'), ('f1', 'f8')]). They are dtypes of structured
    NumPy arrays.

    Parameters
    ----------
    dtype
        A simple dtype or a strategy for simple dtypes for determining the type of
        array elements. If `None`, any supported simple dtype is used.
    allow_array
        Generate only simple dtypes if `False`, else array dtypes as well.

    Examples
    --------
    >>> numpy_dtypes().example()
    dtype(...)
    '''
    if dtype is None:
        dtype = supported_dtypes()
    if not isinstance(dtype, st.SearchStrategy):
        dtype = st.just(dtype)
    if not allow_array:
        return dtype
    return st_np.array_dtypes(subtype_strategy=dtype, allow_subarrays=True)


def numpy_arrays(
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None = None,
    allow_structured: bool = True,
    allow_nan: bool = False,
) -> st.SearchStrategy[np.ndarray]:
    '''Strategy for NumPy arrays from which Awkward Arrays can be created.

    Parameters
    ----------
    dtype
        A simple dtype or a strategy for simple dtypes for determining the type of
        array elements. If `None`, any supported simple dtype is used.
    allow_structured
        Generate only simple arrays if `False`, else structured arrays as well.
    allow_nan
        Generate potentially `NaN` for relevant dtypes if `True`.

    Examples
    --------
    >>> n = numpy_arrays().example()
    >>> ak.from_numpy(n)
    <Array ... type='...'>
    '''

    return st_np.arrays(
        dtype=numpy_dtypes(dtype=dtype, allow_array=allow_structured),
        shape=st_np.array_shapes(),
        elements={'allow_nan': allow_nan},
    )


def from_numpy(
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None = None,
    allow_structured: bool = True,
    allow_nan: bool = False,
) -> st.SearchStrategy[ak.Array]:
    '''Strategy for Awkward Arrays created from NumPy arrays.

    Parameters
    ----------
    dtype
        A simple dtype or a strategy for simple dtypes for determining the type of
        array elements. If `None`, any supported simple dtype is used.
    allow_structured
        Generate only from simple NumPy arrays if `False`, else from structured NumPy
        arrays as well.
    allow_nan
        Generate potentially `NaN` for relevant dtypes if `True`.

    Examples
    --------
    >>> a = from_numpy().example()
    '''
    return st.builds(
        ak.from_numpy,
        numpy_arrays(
            dtype=dtype, allow_structured=allow_structured, allow_nan=allow_nan
        ),
    )
