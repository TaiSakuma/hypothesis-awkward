import numpy as np

import awkward as ak


def any_nan_nat_in_awkward_array(
    a: ak.Array | ak.contents.Content,
    /,
) -> bool:
    '''`True` if Awkward Array contains any `NaN` or `NaT` values, else `False`.

    Parameters
    ----------
    a
        An Awkward Array.

    Returns
    -------
    bool
        `True` if `a` contains any `NaN` or `NaT` values, else `False`.

    Examples
    --------

    >>> a = ak.Array([1.0, 2.0, np.nan])
    >>> any_nan_nat_in_awkward_array(a)
    True

    >>> a = ak.Array([1.0, 2.0, 3.0])
    >>> any_nan_nat_in_awkward_array(a)
    False

    >>> a = ak.Array([{'x': 1.0, 'y': np.nan}, {'x': 2.0, 'y': 3.0}])
    >>> any_nan_nat_in_awkward_array(a)
    True

    '''
    stack: list[ak.Array | ak.contents.Content] = [a]
    while stack:
        item = stack.pop()
        match item:
            case ak.Array():
                stack.append(item.layout)
            case ak.contents.RecordArray():
                for field in item.fields:
                    stack.append(item[field])
            case ak.contents.NumpyArray():
                arr = item.data
                kind = arr.dtype.kind
                if kind in {'f', 'c'} and np.any(np.isnan(arr)):
                    return True
                if kind in {'m', 'M'} and np.any(np.isnat(arr)):
                    return True
            case _:
                raise TypeError(f'Unexpected type: {type(item)}')
    return False
