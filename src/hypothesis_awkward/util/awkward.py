import numpy as np

import awkward as ak


def any_nan_nat_in_awkward_array(
    a: ak.Array | ak.contents.RecordArray | ak.contents.NumpyArray,
    /,
) -> bool:
    match a:
        case ak.Array():
            return any_nan_nat_in_awkward_array(a.layout)
        case ak.contents.RecordArray():
            return any(any_nan_nat_in_awkward_array(a[field]) for field in a.fields)
        case ak.contents.NumpyArray():
            arr = a.data
            kind = arr.dtype.kind
            if kind in {'f', 'c'}:
                return bool(np.any(np.isnan(arr)))
            elif kind in {'m', 'M'}:
                return bool(np.any(np.isnat(arr)))
            else:
                return False
        case _:
            raise TypeError(f'Unexpected type: {type(a)}')
