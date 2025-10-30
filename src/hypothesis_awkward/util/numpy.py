import numpy as np


def any_nan_nat_in_numpy_array(n: np.ndarray) -> bool:
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
