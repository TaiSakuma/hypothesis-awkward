from hypothesis import given
from hypothesis import strategies as st

from hypothesis_awkward.strategies import ranges
from hypothesis_awkward.util import safe_compare


def test_repr() -> None:
    assert repr(safe_compare(1)) == '1'
    assert repr(safe_compare(None)) == 'GreaterAndLessThanAny()'


@given(st.data())
def test_safe_compare(data: st.DataObject) -> None:
    allow_equal = data.draw(st.booleans())
    none_or_small, none_or_large = data.draw(
        ranges(st.integers, allow_equal=allow_equal)
    )
    if allow_equal:
        assert safe_compare(none_or_small) <= safe_compare(none_or_large)
        assert safe_compare(none_or_large) >= safe_compare(none_or_small)
    else:
        assert safe_compare(none_or_small) < safe_compare(none_or_large)
        assert safe_compare(none_or_large) > safe_compare(none_or_small)
