import numpy as np
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import ListOffsetArray, NumpyArray


@st.composite
def string_contents(
    draw: st.DrawFn,
    *,
    alphabet: st.SearchStrategy[str] | None = None,
    min_size: int = 0,
    max_size: int = 10,
) -> ListOffsetArray:
    '''Strategy for ListOffsetArray string content.

    Parameters
    ----------
    alphabet
        A strategy for characters used in the generated strings. If ``None``,
        the full Unicode range is used.
    min_size
        Minimum number of strings.
    max_size
        Maximum number of strings.

    Examples
    --------
    >>> c = string_contents().example()
    >>> isinstance(c, ListOffsetArray)
    True
    '''
    text_st = st.text() if alphabet is None else st.text(alphabet=alphabet)
    strings = draw(st.lists(text_st, min_size=min_size, max_size=max_size))
    encoded = [s.encode('utf-8') for s in strings]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    for i, b in enumerate(encoded):
        offsets[i + 1] = offsets[i] + len(b)
    data = b''.join(encoded)
    content = NumpyArray(
        np.frombuffer(data, dtype=np.uint8) if data else np.array([], dtype=np.uint8),
        parameters={'__array__': 'char'},
    )
    return ListOffsetArray(
        ak.index.Index64(offsets), content, parameters={'__array__': 'string'}
    )
