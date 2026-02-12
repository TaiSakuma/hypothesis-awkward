import string
from functools import partial

from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak


@st.composite
def dicts_for_dataframe(
    draw: st.DrawFn,
    *,
    max_columns: int = 4,
    min_rows: int = 0,
    max_rows: int | None = None,
    allow_none: bool = True,
    allow_list: bool = True,
    allow_nested: bool = True,
    allow_empty: bool = True,
) -> dict[str, list]:
    '''Strategy for dicts for Pandas DataFrame initialization.

    Parameters
    ----------
    max_columns
        Maximum number of keys in the dict.
    min_rows
        Minimum size of each list, a value of the dict. All values are lists of the same
        size.
    max_rows
        Maximum size of each list, a value of the dict. All values are lists of the same
        size.
    allow_none
        List elements may be `None` if `True`.
    allow_list
        List elements may be lists if `True`.
    allow_nested
        When list elements are lists, those lists may be nested lists if `True`.
    allow_empty
        When list elements are lists, those lists may be empty if `True`.

    '''
    if max_rows is None:
        max_rows = max(5, min_rows + 5)

    st_column_names = st.text(alphabet=string.ascii_letters, max_size=3)
    column_names = draw(
        st.lists(st_column_names, min_size=1, max_size=max_columns, unique=True)
    )
    st_leaf_values = st.text(alphabet=string.ascii_letters, max_size=5)
    st_leaves = st_ak.none_or(st_leaf_values) if allow_none else st_leaf_values

    def nest_base(s: st.SearchStrategy) -> st.SearchStrategy:
        min_size = 0 if allow_empty else 1
        return st.lists(s, min_size=min_size, max_size=5)

    def nest(s: st.SearchStrategy) -> st.SearchStrategy:
        return st_ak.none_or(nest_base(s)) if allow_none else nest_base(s)

    if not allow_list:
        st_row_items = st.just(st_leaves)
    elif not allow_nested:
        st_row_items = st.one_of(st.just(st_leaves), st.just(nest(st_leaves)))
    else:
        st_row_items = st.recursive(
            base=st.just(st_leaves),
            extend=lambda s: s.map(nest),
            max_leaves=20,
        )

    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    st_rows = partial(st.lists, min_size=n_rows, max_size=n_rows)
    mapping = {n: st_rows(draw(st_row_items)) for n in column_names}
    return draw(st.fixed_dictionaries(mapping))
