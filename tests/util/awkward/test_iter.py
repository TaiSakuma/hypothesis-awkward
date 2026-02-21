import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import (
    Content,
    EmptyArray,
    ListArray,
    ListOffsetArray,
    NumpyArray,
    RegularArray,
)
from hypothesis_awkward.util import iter_numpy_arrays
from hypothesis_awkward.util.awkward import iter_contents, iter_leaf_contents


@given(data=st.data())
def test_iter_numpy_arrays(data: st.DataObject) -> None:
    '''Verify iter_numpy_arrays yields NumpyArray leaf data.'''
    a = data.draw(st_ak.constructors.arrays(allow_virtual=False), label='array')
    exclude_string = data.draw(st.booleans(), label='exclude_string')
    exclude_bytestring = data.draw(st.booleans(), label='exclude_bytestring')

    result = list(
        iter_numpy_arrays(
            a, exclude_string=exclude_string, exclude_bytestring=exclude_bytestring
        )
    )

    assert all(isinstance(arr, np.ndarray) for arr in result)

    # Top-level NumpyArray data is in result
    result_ids = {id(arr) for arr in result}
    if isinstance(a.layout, NumpyArray):
        assert id(a.layout.data) in result_ids

    # NumpyArray children of list-type arrays are in result
    for c in iter_contents(
        a, string_as_leaf=exclude_string, bytestring_as_leaf=exclude_bytestring
    ):
        if not isinstance(c, (ListOffsetArray, ListArray, RegularArray)):
            continue
        if exclude_string and c.parameter('__array__') == 'string':
            continue
        if exclude_bytestring and c.parameter('__array__') == 'bytestring':
            continue
        if isinstance(c.content, NumpyArray):
            assert id(c.content.data) in result_ids


@given(data=st.data())
def test_iter_leaf_contents(data: st.DataObject) -> None:
    '''Verify all yielded items are leaf content types.'''
    a = data.draw(st_ak.constructors.arrays(), label='array')
    string_as_leaf = data.draw(st.booleans(), label='string_as_leaf')
    bytestring_as_leaf = data.draw(st.booleans(), label='bytestring_as_leaf')
    for content in iter_leaf_contents(
        a, string_as_leaf=string_as_leaf, bytestring_as_leaf=bytestring_as_leaf
    ):
        assert isinstance(
            content,
            (NumpyArray, EmptyArray, ListOffsetArray, ListArray, RegularArray),
        )
        if string_as_leaf:
            assert content.parameter('__array__') != 'char'
        else:
            assert content.parameter('__array__') != 'string'
        if bytestring_as_leaf:
            assert content.parameter('__array__') != 'byte'
        else:
            assert content.parameter('__array__') != 'bytestring'


def _children(c: Content) -> list[Content]:
    '''Return direct Content children of a node.'''
    match c:
        case ak.contents.RecordArray():
            return list(c.contents)
        case ak.contents.UnionArray():
            return list(c.contents)
        case _ if hasattr(c, 'content'):
            return [c.content]
        case _:
            return []


@given(data=st.data())
def test_iter_contents(data: st.DataObject) -> None:
    '''Verify iter_contents yields exactly the full Content tree.'''
    a = data.draw(st_ak.constructors.arrays(), label='array')
    string_as_leaf = data.draw(st.booleans(), label='string_as_leaf')
    bytestring_as_leaf = data.draw(st.booleans(), label='bytestring_as_leaf')
    all_contents = list(
        iter_contents(
            a, string_as_leaf=string_as_leaf, bytestring_as_leaf=bytestring_as_leaf
        )
    )
    id_set = {id(c) for c in all_contents}

    # 1. Type invariant
    assert all(isinstance(c, Content) for c in all_contents)

    # 2. Root inclusion
    assert id(a.layout) in id_set

    # 3. Closure: children of every yielded node are also yielded
    #    (string/bytestring leaves don't descend, so skip their children)
    for c in all_contents:
        if string_as_leaf and c.parameter('__array__') == 'string':
            continue
        if bytestring_as_leaf and c.parameter('__array__') == 'bytestring':
            continue
        for child in _children(c):
            assert id(child) in id_set

    # 4. No duplicates
    assert len(id_set) == len(all_contents)

    # 5. String/bytestring parameter invariants
    for c in all_contents:
        if string_as_leaf:
            assert c.parameter('__array__') != 'char'
        if bytestring_as_leaf:
            assert c.parameter('__array__') != 'byte'
