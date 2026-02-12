import numpy as np
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward.util import SUPPORTED_DTYPES


def _inner_shapes(
    max_ndim: int = 3,
    max_side: int = 10,
) -> st.SearchStrategy[tuple[int, ...]]:
    '''Strategy for inner_shape tuples.

    Generates empty tuples (common) and small non-empty tuples (rare).
    '''
    return st.one_of(
        st.just(()),
        st.lists(
            st.integers(min_value=1, max_value=max_side),
            min_size=1,
            max_size=max_ndim,
        ).map(tuple),
    )


def numpy_forms(
    type_: ak.types.NumpyType | st.SearchStrategy[ak.types.NumpyType] | None = None,
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_datetime: bool = True,
    inner_shape: tuple[int, ...] | st.SearchStrategy[tuple[int, ...]] | None = None,
    allow_inner_shape: bool = True,
) -> st.SearchStrategy[ak.forms.NumpyForm]:
    '''Strategy for NumpyForm (primitive/leaf forms).

    Parameters
    ----------
    type_
        A NumpyType or strategy for NumpyType. If given, generates a
        NumpyForm that corresponds to the type. When set, `dtypes`,
        `allow_datetime`, `inner_shape`, and `allow_inner_shape` are
        ignored (the form is fully determined by the type).
    dtypes
        Strategy for NumPy dtypes to use as the primitive. If None, uses
        `supported_dtypes()`. Ignored when `type_` is given.
    allow_datetime
        Include datetime64/timedelta64 when `dtypes` is None. Ignored
        when `type_` is given.
    inner_shape
        A fixed inner_shape tuple or a strategy for inner_shape tuples.
        If None, generates inner_shape values controlled by
        `allow_inner_shape`. Ignored when `type_` is given.
    allow_inner_shape
        When `inner_shape` is None, allow generation of non-empty
        inner_shape tuples. If False, inner_shape is always `()`. Only
        effective when `inner_shape` is None. Ignored when `type_` is
        given.

    Examples
    --------
    >>> import hypothesis_awkward.strategies as st_ak
    >>> f = st_ak.numpy_forms().example()
    >>> isinstance(f, ak.forms.NumpyForm)
    True
    '''
    # Type mode: form is fully determined by the NumpyType
    if type_ is not None:
        if isinstance(type_, ak.types.NumpyType):
            type_ = st.just(type_)
        return type_.map(lambda t: ak.forms.NumpyForm(t.primitive))

    # Dtypes mode: generate from dtypes and inner_shape
    if dtypes is None:
        if allow_datetime:
            dtypes = st.sampled_from(SUPPORTED_DTYPES)
        else:
            dtypes = st.sampled_from(
                tuple(d for d in SUPPORTED_DTYPES if d.kind not in ('M', 'm'))
            )

    st_primitive = dtypes.map(lambda d: d.name)

    if inner_shape is not None:
        if isinstance(inner_shape, tuple):
            inner_shape = st.just(inner_shape)
        st_shape = inner_shape
    elif allow_inner_shape:
        st_shape = _inner_shapes()
    else:
        st_shape = st.just(())

    return st.tuples(st_primitive, st_shape).map(
        lambda args: ak.forms.NumpyForm(args[0], args[1])
    )
