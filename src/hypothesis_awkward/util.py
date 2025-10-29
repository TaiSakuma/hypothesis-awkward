import numpy as np


def simple_dtype_kinds_in(d: np.dtype) -> set[str]:
    '''Kinds of simple dtypes (e.g. `i`, `f`, `M`) contained in `d`.'''
    match d.names, d.kind, d.subdtype:
        case None, kind, None:  # simple dtype
            return {kind}
        case None, 'V', tuple(subdtype):  # sub-array dtype
            # E.g., subdtype == (dtype('int32'), (3, 4))
            return {subdtype[0].kind}
        case tuple(names), 'V', None:  # structured dtype
            kinds = set()
            for name in names:
                f = d.fields
                assert f is not None
                kinds.update(simple_dtype_kinds_in(f[name][0]))
            return kinds
        case _:
            raise TypeError(f'Unexpected dtype: {d}')
