from typing import TypedDict

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak


class FromListsKwargs(TypedDict, total=False):
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None
    allow_nan: bool
    max_size: int
    max_depth: int


@st.composite
def from_lists_kwargs(draw: st.DrawFn) -> FromListsKwargs:
    kwargs = FromListsKwargs()

    if draw(st.booleans()):
        kwargs['dtype'] = draw(
            st.one_of(
                st.none(),
                st.just(st_ak.builtin_safe_dtypes()),
                st_ak.builtin_safe_dtypes(),
            )
        )

    if draw(st.booleans()):
        kwargs['allow_nan'] = draw(st.booleans())

    if draw(st.booleans()):
        kwargs['max_size'] = draw(st.integers(min_value=0, max_value=5))

    if draw(st.booleans()):
        kwargs['max_depth'] = draw(st.integers(min_value=1, max_value=5))

    return kwargs


@given(data=st.data())
def test_from_list(data: st.DataObject) -> None:
    # Draw options
    kwargs = data.draw(from_lists_kwargs(), label='kwargs')

    # Call the test subject
    a = data.draw(st_ak.from_list(**kwargs), label='a')
    assert isinstance(a, ak.Array)
