from hypothesis import given
from hypothesis import strategies as st

from hypothesis_awkward.util import safe_max, safe_min


@given(st.data())
def test_safe_min(data: st.DataObject) -> None:
    int_vals = data.draw(st.lists(st.integers()))
    none_vals = data.draw(st.lists(st.none()))
    vals = data.draw(st.permutations(int_vals + none_vals))  # type: ignore

    default_val = data.draw(st.one_of(st.integers(), st.none()))

    result = safe_min(vals, default=default_val)

    if int_vals:
        assert result == min(int_vals)
    else:
        assert result == default_val


@given(st.data())
def test_safe_max(data: st.DataObject) -> None:
    int_vals = data.draw(st.lists(st.integers()))
    none_vals = data.draw(st.lists(st.none()))
    vals = data.draw(st.permutations(int_vals + none_vals))  # type: ignore

    default_val = data.draw(st.one_of(st.integers(), st.none()))

    result = safe_max(vals, default=default_val)

    if int_vals:
        assert result == max(int_vals)
    else:
        assert result == default_val
