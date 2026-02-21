import numpy as np
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import ListOffsetArray, NumpyArray


@st.composite
def bytestring_contents(
    draw: st.DrawFn,
    *,
    min_size: int = 0,
    max_size: int = 10,
) -> ListOffsetArray:
    '''Strategy for ListOffsetArray bytestring content.

    Parameters
    ----------
    min_size
        Minimum number of bytestrings.
    max_size
        Maximum number of bytestrings.

    Examples
    --------
    >>> c = bytestring_contents().example()
    >>> isinstance(c, ListOffsetArray)
    True
    '''
    bytestrings = draw(st.lists(st.binary(), min_size=min_size, max_size=max_size))
    offsets = np.zeros(len(bytestrings) + 1, dtype=np.int64)
    for i, b in enumerate(bytestrings):
        offsets[i + 1] = offsets[i] + len(b)
    data = b''.join(bytestrings)
    content = NumpyArray(
        np.frombuffer(data, dtype=np.uint8) if data else np.array([], dtype=np.uint8),
        parameters={'__array__': 'byte'},
    )
    return ListOffsetArray(
        ak.index.Index64(offsets), content, parameters={'__array__': 'bytestring'}
    )
