from typing import TypedDict, cast

import pytest
from hypothesis import Phase, find, given, note, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak

pytestmark = pytest.mark.skip(reason='draft — content strategy testing TBD')


class ListTypesKwargs(TypedDict, total=False):
    '''Options for `list_types()` strategy.'''

    content: st.SearchStrategy[ak.types.Type] | None


def list_types_kwargs() -> st.SearchStrategy[ListTypesKwargs]:
    '''Strategy for options for `list_types()` strategy.'''
    return st.fixed_dictionaries(
        {},
        optional={
            'content': st.one_of(
                st.none(),
                st.just(st_ak.numpy_types()),
            ),
        },
    ).map(lambda d: cast(ListTypesKwargs, d))


@settings(max_examples=200)
@given(data=st.data())
def test_list_types(data: st.DataObject) -> None:
    '''Test that `list_types()` respects all its options.'''
    # Draw options
    kwargs = data.draw(list_types_kwargs(), label='kwargs')

    # Call the test subject
    result = data.draw(st_ak.list_types(**kwargs), label='result')

    # Assert the result is a ListType
    assert isinstance(result, ak.types.ListType)

    # Assert the options were effective
    content = kwargs.get('content', None)

    note(f'{result=}')
    note(f'{result.content=}')

    # The content should be a valid Type
    assert isinstance(result.content, ak.types.Type)

    # If content is None (default), content should be NumpyType
    if content is None:
        assert isinstance(result.content, ak.types.NumpyType)


def test_draw_list_of_numpy() -> None:
    '''Assert that ListType with NumpyType content can be drawn.'''
    find(
        st_ak.list_types(),
        lambda t: isinstance(t.content, ak.types.NumpyType),
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_with_custom_content() -> None:
    '''Assert that custom content strategy is respected.'''
    # Use a specific content type
    int64_content = st.just(ak.types.NumpyType('int64'))

    find(
        st_ak.list_types(content=int64_content),
        lambda t: isinstance(t.content, ak.types.NumpyType)
        and t.content.primitive == 'int64',
        settings=settings(phases=[Phase.generate]),
    )
