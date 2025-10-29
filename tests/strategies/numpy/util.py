import numpy as np


def _dtype_kinds(d: np.dtype) -> set[str]:
    '''Kinds of simple dtypes (e.g. `i`, `f`, `M`) contained in `d`.'''
    if d.names is None:  # simple dtype
        kind = d.kind
        if kind == 'V' and d.subdtype is not None:
            kind = d.subdtype[0].kind
        return {kind}
    else:  # structured dtype
        kinds = set()
        for name in d.names:
            f = d.fields
            assert f is not None
            kinds.update(_dtype_kinds(f[name][0]))
        return kinds
