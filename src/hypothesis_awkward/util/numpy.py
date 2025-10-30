import numpy as np


def any_nan_nat_in_numpy_array(n: np.ndarray, /) -> bool:
    '''`True` if NumPy array contains any `NaN` or `NaT` values, else `False`.

    Parameters
    ----------
    n
        A NumPy array.

    Returns
    -------
    bool
        `True` if `n` contains any `NaN` or `NaT` values, else `False`.

    Examples
    --------

    >>> n = np.array([1.0, 2.0, np.nan])
    >>> any_nan_nat_in_numpy_array(n)
    True

    >>> n = np.array([1.0, 2.0, 3.0])
    >>> any_nan_nat_in_numpy_array(n)
    False

    >>> n = np.array([(1, np.datetime64('2020-01-01')),
    ...               (2, np.datetime64('NaT'))],
    ...              dtype=[('a', 'i4'), ('b', 'M8[D]')])
    >>> any_nan_nat_in_numpy_array(n)
    True

    '''

    kind = n.dtype.kind
    match kind:
        case 'V':  # structured
            return any(any_nan_nat_in_numpy_array(n[field]) for field in n.dtype.names)
        case 'f' | 'c':  # float or complex
            return bool(np.any(np.isnan(n)))
        case 'm' | 'M':  # timedelta or datetime
            return bool(np.any(np.isnat(n)))
        case _:
            return False
