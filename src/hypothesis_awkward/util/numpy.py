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
    return any_nan_in_numpy_array(n) or any_nat_in_numpy_array(n)


def any_nan_in_numpy_array(n: np.ndarray, /) -> bool:
    '''`True` if NumPy array contains any `NaN` values, else `False`.

    Parameters
    ----------
    n
        A NumPy array.

    Returns
    -------
    bool
        `True` if `n` contains any `NaN` values, else `False`.

    Examples
    --------

    >>> n = np.array([1.0, 2.0, np.nan])
    >>> any_nan_in_numpy_array(n)
    True

    >>> n = np.array([1.0, 2.0, 3.0])
    >>> any_nan_in_numpy_array(n)
    False

    >>> n = np.array([(1, 2.0), (2, np.nan)],
    ...              dtype=[('a', 'i4'), ('b', 'f8')])
    >>> any_nan_in_numpy_array(n)
    True

    '''
    stack = [n]
    while stack:
        arr = stack.pop()
        match arr.dtype.kind:
            case 'V':  # structured
                stack.extend(arr[field] for field in arr.dtype.names)
            case 'f' | 'c':  # float or complex
                if np.any(np.isnan(arr)):
                    return True
    return False


def any_nat_in_numpy_array(n: np.ndarray, /) -> bool:
    '''`True` if NumPy array contains any `NaT` values, else `False`.

    Parameters
    ----------
    n
        A NumPy array.

    Returns
    -------
    bool
        `True` if `n` contains any `NaT` values, else `False`.

    Examples
    --------

    >>> n = np.array(['2020-01-01', 'NaT'], dtype='datetime64[D]')
    >>> any_nat_in_numpy_array(n)
    True

    >>> n = np.array(['2020-01-01', '2020-01-02'], dtype='datetime64[D]')
    >>> any_nat_in_numpy_array(n)
    False

    >>> n = np.array([(1, np.datetime64('2020-01-01')),
    ...               (2, np.datetime64('NaT'))],
    ...              dtype=[('a', 'i4'), ('b', 'M8[D]')])
    >>> any_nat_in_numpy_array(n)
    True

    '''
    stack = [n]
    while stack:
        arr = stack.pop()
        match arr.dtype.kind:
            case 'V':  # structured
                stack.extend(arr[field] for field in arr.dtype.names)
            case 'm' | 'M':  # timedelta or datetime
                if np.any(np.isnat(arr)):
                    return True
    return False
