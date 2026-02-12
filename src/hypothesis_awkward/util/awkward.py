from collections.abc import Iterator

import numpy as np

import awkward as ak
from awkward.contents import (
    Content,
    EmptyArray,
    IndexedOptionArray,
    ListArray,
    ListOffsetArray,
    NumpyArray,
    RecordArray,
    RegularArray,
    UnionArray,
    UnmaskedArray,
)


def any_nan_nat_in_awkward_array(a: ak.Array | Content, /) -> bool:
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
    return any_nan_in_awkward_array(a) or any_nat_in_awkward_array(a)


def any_nan_in_awkward_array(a: ak.Array | Content, /) -> bool:
    '''`True` if Awkward Array contains any `NaN` values, else `False`.

    Parameters
    ----------
    a
        An Awkward Array.

    Returns
    -------
    bool
        `True` if `a` contains any `NaN` values, else `False`.

    Examples
    --------

    >>> a = ak.Array([1.0, 2.0, np.nan])
    >>> any_nan_in_awkward_array(a)
    True

    >>> a = ak.Array([1.0, 2.0, 3.0])
    >>> any_nan_in_awkward_array(a)
    False

    >>> a = ak.Array([{'x': 1.0, 'y': np.nan}, {'x': 2.0, 'y': 3.0}])
    >>> any_nan_in_awkward_array(a)
    True

    '''
    for arr in iter_numpy_arrays(a):
        if arr.dtype.kind in {'f', 'c'} and np.any(np.isnan(arr)):
            return True
    return False


def any_nat_in_awkward_array(a: ak.Array | Content, /) -> bool:
    '''`True` if Awkward Array contains any `NaT` values, else `False`.

    Parameters
    ----------
    a
        An Awkward Array.

    Returns
    -------
    bool
        `True` if `a` contains any `NaT` values, else `False`.

    Examples
    --------

    >>> a = ak.Array(np.array(['2020-01-01', 'NaT'], dtype='datetime64[D]'))
    >>> any_nat_in_awkward_array(a)
    True

    >>> a = ak.Array(np.array(['2020-01-01', '2020-01-02'], dtype='datetime64[D]'))
    >>> any_nat_in_awkward_array(a)
    False

    '''
    for arr in iter_numpy_arrays(a):
        if arr.dtype.kind in {'m', 'M'} and np.any(np.isnat(arr)):
            return True
    return False


def iter_contents(a: ak.Array | Content, /) -> Iterator[Content]:
    '''Iterate over all contents in an Awkward Array layout.

    Parameters
    ----------
    a
        An Awkward Array or Content.

    Yields
    ------
    Content
        Each content node in the layout.

    '''
    stack: list[ak.Array | Content] = [a]
    while stack:
        item = stack.pop()
        match item:
            case ak.Array():
                stack.append(item.layout)
            case NumpyArray() | EmptyArray():
                yield item
            case RecordArray():
                yield item
                stack.extend([item[f] for f in item.fields])
            case (
                IndexedOptionArray()
                | ListArray()
                | ListOffsetArray()
                | RegularArray()
                | UnmaskedArray()
            ):
                yield item
                stack.append(item.content)
            case UnionArray():
                yield item
                stack.extend(item.contents)
            case _:  # pragma: no cover
                raise TypeError(f'Unexpected content type: {type(item)}')


def iter_leaf_contents(a: ak.Array | Content, /) -> Iterator[EmptyArray | NumpyArray]:
    '''Iterate over all leaf contents in an Awkward Array layout.

    Parameters
    ----------
    a
        An Awkward Array or Content.

    Yields
    ------
    EmptyArray | NumpyArray
        Each leaf content in the layout.

    '''
    for content in iter_contents(a):
        if isinstance(content, (EmptyArray, NumpyArray)):
            yield content


def iter_numpy_arrays(a: ak.Array | Content, /) -> Iterator[np.ndarray]:
    '''Iterate over all NumPy arrays in an Awkward Array layout.

    Parameters
    ----------
    a
        An Awkward Array or Content.

    Yields
    ------
    np.ndarray
        Each underlying NumPy array in the layout.

    Examples
    --------

    >>> a = ak.Array([[1.0, 2.0], [3.0]])
    >>> list(iter_numpy_arrays(a))
    [array([1., 2., 3.])]

    >>> a = ak.Array([{'x': 1, 'y': 2.0}, {'x': 3, 'y': 4.0}])
    >>> sorted([arr.dtype for arr in iter_numpy_arrays(a)], key=str)
    [dtype('float64'), dtype('int64')]

    '''
    for content in iter_leaf_contents(a):
        if isinstance(content, NumpyArray):
            yield content.data
