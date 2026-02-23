import numpy as np
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak


@st.composite
def arrays(
    draw: st.DrawFn,
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    max_size: int = 10,
    allow_nan: bool = False,
    allow_numpy: bool = True,
    allow_empty: bool = True,
    allow_string: bool = True,
    allow_bytestring: bool = True,
    allow_regular: bool = True,
    allow_list_offset: bool = True,
    allow_list: bool = True,
    allow_record: bool = True,
    allow_union: bool = True,
    max_depth: int = 5,
    max_length: int | None = None,
    allow_virtual: bool = True,
) -> ak.Array:
    '''Strategy for Awkward Arrays.

    Builds arrays with NumpyArray, EmptyArray, string, and bytestring as leaf
    contents that can be nested multiple levels deep in RegularArray,
    ListOffsetArray, ListArray, RecordArray, and UnionArray. Arrays might be
    virtual.

    Parameters
    ----------
    dtypes
        A strategy for NumPy scalar dtypes used in ``NumpyArray``. If ``None``, the
        default strategy that generates any scalar dtype supported by Awkward Array is
        used. Does not affect string or bytestring content.
    max_size
        Maximum total number of elements in the generated content. Each
        numerical value, including complex and datetime, counts as one. Each
        string and bytestring (not character or byte) counts as one.
    allow_nan
        No ``NaN``/``NaT`` values are generated in ``NumpyArray`` if ``False``.
    allow_numpy
        No ``NumpyArray`` is generated if ``False``.
    allow_empty
        No ``EmptyArray`` is generated if ``False``. ``EmptyArray`` has Awkward
        type ``unknown`` and carries no data. Unlike ``NumpyArray``, it is
        unaffected by ``dtypes`` and ``allow_nan``.
    allow_string
        No string content is generated if ``False``. Strings are represented
        as a ``ListOffsetArray`` wrapping a ``NumpyArray(uint8)``. Each
        string (not character) counts toward ``max_size``. The string
        itself does not count toward ``max_depth``. Unaffected by ``dtypes``
        and ``allow_nan``.
    allow_bytestring
        No bytestring content is generated if ``False``. Bytestrings are
        represented as a ``ListOffsetArray`` wrapping a ``NumpyArray(uint8)``.
        Each bytestring (not byte) counts toward ``max_size``. The
        bytestring itself does not count toward ``max_depth``. Unaffected
        by ``dtypes`` and ``allow_nan``.
    allow_regular
        No ``RegularArray`` is generated if ``False``.
    allow_list_offset
        No ``ListOffsetArray`` is generated if ``False``.
    allow_list
        No ``ListArray`` is generated if ``False``.
    allow_record
        No ``RecordArray`` is generated if ``False``.
    allow_union
        No ``UnionArray`` is generated if ``False``.
    max_depth
        Maximum nesting depth. Each RegularArray, ListOffsetArray, ListArray,
        RecordArray, and UnionArray layer adds one level, excluding those that
        form string or bytestring content.
    max_length
        Maximum ``len()`` of the generated array. No constraint when ``None`` (the
        default).
    allow_virtual
        No virtual arrays are generated if ``False``.

    Examples
    --------
    >>> arrays().example()
    <Array ... type='...'>

    '''
    layout = draw(
        st_ak.contents.contents(
            dtypes=dtypes,
            max_size=max_size,
            allow_nan=allow_nan,
            allow_numpy=allow_numpy,
            allow_empty=allow_empty,
            allow_regular=allow_regular,
            allow_list_offset=allow_list_offset,
            allow_list=allow_list,
            allow_string=allow_string,
            allow_bytestring=allow_bytestring,
            allow_record=allow_record,
            allow_union=allow_union,
            max_depth=max_depth,
            max_length=max_length,
        )
    )
    array = ak.Array(layout)
    to_lazify = allow_virtual and draw(st.booleans())
    if not to_lazify:
        return array
    return _lazify(array)


def _lazify(array: ak.Array) -> ak.Array:
    form, length, buffers = ak.to_buffers(array)
    if not buffers:
        return array
    virtual_buffers = {k: (lambda v=v: v) for k, v in buffers.items()}
    return ak.from_buffers(form, length, virtual_buffers)
