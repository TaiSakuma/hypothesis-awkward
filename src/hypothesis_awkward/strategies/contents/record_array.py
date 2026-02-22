import string

from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, RecordArray


@st.composite
def record_array_contents(
    draw: st.DrawFn,
    contents: list[Content] | st.SearchStrategy[list[Content]] | None = None,
    *,
    max_fields: int = 5,
    allow_tuple: bool = True,
) -> Content:
    '''Strategy for RecordArray Content from a list of child Contents.

    Parameters
    ----------
    contents
        Child contents. Can be a strategy for a list of Content, a concrete
        list, or ``None`` to draw random children.
    max_fields
        Maximum number of fields when ``contents`` is ``None``.
    allow_tuple
        Allow tuple records (no field names) if ``True``.

    Examples
    --------
    >>> c = record_array_contents().example()
    >>> isinstance(c, Content)
    True
    '''
    match contents:
        case None:
            contents = draw(st_ak.contents.content_lists(max_size=max_fields))
        case st.SearchStrategy():
            contents = draw(contents)
        case list():
            pass
    assert isinstance(contents, list)

    if allow_tuple:
        is_tuple = draw(st.booleans())
    else:
        is_tuple = False

    if is_tuple:
        fields = None
    else:
        st_names = st.text(alphabet=string.ascii_letters, max_size=3)
        fields = draw(
            st.lists(
                st_names,
                min_size=len(contents),
                max_size=len(contents),
                unique=True,
            )
        )

    length = 0 if not contents else None
    return RecordArray(contents, fields=fields, length=length)
