'''Tests for `RecordCallDraws` and `OptsChain.register_callable()`.'''

from hypothesis import given, settings
from hypothesis import strategies as st

from hypothesis_awkward.strategies.misc.record import RecordCallDraws, RecordDraws


@settings(max_examples=200)
@given(data=st.data())
def test_record_call_draws(data: st.DataObject) -> None:
    '''Calling a RecordCallDraws records every drawn value in order.'''
    recorder = RecordCallDraws(st.integers)

    n_resets = data.draw(st.integers(min_value=0, max_value=3), label='n_resets')
    for r in range(n_resets + 1):
        recorder.reset()
        expected: list[int] = []
        n_calls = data.draw(st.integers(min_value=1, max_value=5), label=f'n_calls_{r}')
        for i in range(n_calls):
            max_value = data.draw(
                st.integers(min_value=0, max_value=100),
                label=f'max_value_{r}_{i}',
            )
            wrapped = recorder(max_value=max_value)
            assert isinstance(wrapped, RecordDraws)
            value = data.draw(wrapped, label=f'value_{r}_{i}')
            assert value <= max_value
            expected.append(value)

        assert recorder.drawn == expected
