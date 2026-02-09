from collections.abc import Mapping

import numpy as np

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


def simple_dtypes_in(d: np.dtype, /) -> set[np.dtype]:
    '''Return simple dtypes contained in a (compound) dtype `d`.

    Parameters
    ----------
    d
        A NumPy dtype. It can be a sub-array or structured dtype as well as a simple
        dtype.

    Returns
    -------
    set of np.dtype
        Simple dtypes contained in `d`.

    Examples
    --------
    >>> simple_dtypes_in(np.dtype('int32'))
    {dtype('int32')}

    >>> sorted(simple_dtypes_in(np.dtype([('f0', 'i4'), ('f1', 'f8')])))
    [dtype('int32'), dtype('float64')]

    '''

    match d.names, d.kind, d.subdtype, d.fields:
        case None, str(), None, None:
            # Simple dtype
            # e.g., d = dtype('int32')
            return {d}
        case None, 'V', tuple(subdtype), None:
            # Sub-array dtype
            # e.g., d = dtype(('int32', (3, 4)))
            return simple_dtypes_in(subdtype[0])
        case tuple(names), 'V', None, fields if isinstance(fields, Mapping):
            # Structured dtype
            # e.g., d = dtype([('f0', 'i4'), ('f1', 'f8')])
            return {t for n in names for t in simple_dtypes_in(fields[n][0])}
        case _:  # pragma: no cover
            raise TypeError(f'Unexpected dtype: {d}')


def simple_dtype_kinds_in(d: np.dtype, /) -> set[str]:
    '''Return character codes of simple dtypes contained in a (compound) dtype `d`.

    Parameters
    ----------
    d
        A NumPy dtype. It can be a sub-array or structured dtype as well as a simple
        dtype.

    Returns
    -------
    set of str
        Character codes of simple dtypes contained in `d`.

    Examples
    --------
    >>> simple_dtype_kinds_in(np.dtype('int32'))
    {'i'}

    >>> sorted(simple_dtype_kinds_in(np.dtype([('f0', 'i4'), ('f1', 'f8')])))
    ['f', 'i']
    '''

    return {t.kind for t in simple_dtypes_in(d)}


def n_scalars_in(d: np.dtype, /) -> int:
    '''Return the number of scalar values contained in a value of dtype `d`.

    Parameters
    ----------
    d
        A NumPy dtype. It can be a sub-array or structured dtype as well as a simple
        dtype.

    Returns
    -------
    int
        The number of scalar values contained in a value of dtype `d`.

    Examples
    --------
    >>> n_scalars_in(np.dtype('int32'))
    1

    >>> n_scalars_in(np.dtype(('int32', (3, 4))))
    12

    >>> n_scalars_in(np.dtype([('f0', 'i4'), ('f1', ('f8', (2,)))]))
    3
    '''

    match d.names, d.kind, d.subdtype, d.fields:
        case None, str(), None, None:
            # Simple dtype
            # e.g., d = dtype('int32')
            return 1
        case None, 'V', tuple(subdtype), None:
            # Sub-array dtype
            # e.g., d = dtype(('int32', (3, 4)))
            return int(np.prod(subdtype[1])) * n_scalars_in(subdtype[0])
        case tuple(names), 'V', None, fields if isinstance(fields, Mapping):
            # Structured dtype
            # e.g., d = dtype([('f0', 'i4'), ('f1', ('f8', (2,)))])
            return sum(n_scalars_in(fields[n][0]) for n in names)
        case _:  # pragma: no cover
            raise TypeError(f'Unexpected dtype: {d}')
