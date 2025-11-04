from collections.abc import Iterable
from typing import Any, TypedDict, cast

import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak


class DictsForDataFrameKwargs(TypedDict, total=False):
    '''Options for `dicts_for_dataframe()` strategy.'''

    max_columns: int
    max_rows: int
    allow_none: bool
    allow_list: bool
    allow_nested: bool
    allow_empty: bool


def dicts_for_dataframe_kwargs() -> st.SearchStrategy[DictsForDataFrameKwargs]:
    '''Strategy for options for `dicts_for_dataframe()` strategy.'''
    return st.fixed_dictionaries(
        {},
        optional={
            'max_columns': st.integers(min_value=1, max_value=6),
            'max_rows': st.integers(min_value=0, max_value=6),
            'allow_none': st.booleans(),
            'allow_list': st.booleans(),
            'allow_nested': st.booleans(),
            'allow_empty': st.booleans(),
        },
    ).map(lambda d: cast(DictsForDataFrameKwargs, d))


@given(data=st.data())
def test_dicts_for_dataframe(data: st.DataObject) -> None:
    # Draw options
    kwargs = data.draw(dicts_for_dataframe_kwargs(), label='kwargs')

    # Call the test subject
    dict_ = data.draw(st_ak.dicts_for_dataframe(**kwargs), label='dict_')

    # Assert the options were effective
    max_columns = kwargs.get('max_columns', 4)
    max_rows = kwargs.get('max_rows', 5)
    allow_none = kwargs.get('allow_none', True)
    allow_list = kwargs.get('allow_list', True)
    allow_nested = kwargs.get('allow_nested', True)
    allow_empty = kwargs.get('allow_empty', True)

    def _has_none(x: Iterable[list | None | Any]) -> bool:
        stack = list(x)
        while stack:
            v = stack.pop()
            if isinstance(v, list):
                stack.extend(v)
            else:
                if v is None:
                    return True
        return False

    def _has_empty_list(x: list) -> bool:
        stack = list(x)
        while stack:
            v = stack.pop()
            if isinstance(v, list):
                if len(v) == 0:
                    return True
                stack.extend(v)
        return False

    assert len(dict_) <= max_columns

    n_rows = {len(r) for r in dict_.values()}
    assert len(n_rows) <= 1
    if n_rows:
        assert n_rows.pop() <= max_rows

    if not allow_none:
        assert not _has_none(dict_.values())

    if not allow_list:
        for col in dict_.values():
            assert not any(isinstance(v, list) for v in col)

    if not allow_nested:
        for col in dict_.values():
            for v in col:
                if isinstance(v, list):
                    assert not any(isinstance(x, list) for x in v)

    if not allow_empty:
        for col in dict_.values():
            assert not _has_empty_list(col)

    # Assert a Pandas DataFrame can be created
    pd.DataFrame(dict_)

    # Assert an Awkward Array can be created
    a = ak.Array(dict_)
    ak.to_dataframe(a)
