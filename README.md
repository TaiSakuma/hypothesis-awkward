# hypothesis-awkward

_Hypothesis strategies for Awkward Arrays._

[![pypi-badge]][pypi] [![pypi-python-badge]][pypi]

[![test-badge]][test] [![codecov-badge]][codecov]

[pypi-badge]: https://img.shields.io/pypi/v/hypothesis-awkward.svg
[pypi-python-badge]: https://img.shields.io/pypi/pyversions/hypothesis-awkward.svg
[pypi]: https://pypi.org/project/hypothesis-awkward
[test-badge]: https://github.com/TaiSakuma/hypothesis-awkward/actions/workflows/unit-test.yml/badge.svg
[test]: https://github.com/TaiSakuma/hypothesis-awkward/actions/workflows/unit-test.yml
[codecov-badge]: https://codecov.io/gh/TaiSakuma/hypothesis-awkward/graph/badge.svg?token=cffic9D2b3
[codecov]: https://codecov.io/gh/TaiSakuma/hypothesis-awkward

[Hypothesis] is a _property-based testing_ library. Its [_strategies_][hyp-st]
are Python functions that strategically generate test data that can fail test
cases in _pytest_ and other testing frameworks. Once a test fails, Hypothesis
searches for the simplest sample that causes the same error. Hypothesis
automatically explores edge cases; developers do not need to craft test data
manually.

Property-based testing is useful for finding edge cases in _array_ libraries and
in code that uses them. In fact, Hypothesis strategies for [NumPy][hyp-st-numpy]
and [pandas][hyp-st-pandas] data types are included in Hypothesis itself.
[Xarray] provides [strategies for its data structure][xarray-st]. The _Apache
Arrow_ codebase has [strategies for PyArrow][pyarrow-st], which are not
officially documented in its API reference.

This package, [hypothesis-awkward], is a collection of Hypothesis strategies for
[Awkward Array][awkward-array], which can represent a wide variety of layouts of
nested, variable-length, and mixed-type data. The current version of this
package includes strategies that generate samples with certain types of layouts.
The goal is to develop strategies that can generate fully general Awkward Arrays
with multiple options to control the layout, data types, missing values, masks,
and other array attributes. These strategies can help close in on edge cases in
tools that use Awkward Array, and Awkward Array itself.

[hypothesis]: https://github.com/HypothesisWorks/hypothesis
[hyp-st]: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html
[hyp-st-numpy]: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#numpy
[hyp-st-pandas]: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#pandas
[xarray]: https://xarray.dev/
[xarray-st]: https://docs.xarray.dev/en/stable/user-guide/testing.html#hypothesis-testing
[pyarrow-st]: https://github.com/apache/arrow/blob/apache-arrow-22.0.0/python/pyarrow/tests/strategies.py
[hypothesis-awkward]: https://github.com/TaiSakuma/hypothesis-awkward
[awkward-array]: https://awkward-array.org/

> [!NOTE]
>
> This package is early work in progress and still experimental. The APIs may
> change over time.

## Installation

You can install the package from PyPI using pip:

```bash
pip install hypothesis-awkward
```

This also installs Hypothesis and Awkward Array as dependencies unless they are
already installed.

## The strategy `arrays()`

The function `arrays()` is the main strategy. It is currently experimental and
developed in `strategies/constructors/`. The plan is to have `arrays()` generate
fully general Awkward Arrays with many options to control the output arrays.

### Sample outputs of `arrays()`

You can see sample outputs of the current version of `arrays()` in the test
case:

```python
from hypothesis import given

import awkward as ak
import hypothesis_awkward.strategies as st_ak


@given(array=st_ak.constructors.arrays())
def test_array(array: ak.Array) -> None:
    print(f'{array=!r}')
```

For example, this might print:

```python
array=<Array [] type='0 * bool'>
array=<Array [0] type='1 * int16'>
array=<Array [1.72e-11, -3.4e+38, -3.4e+38, -4.05e+15] type='4 * float32'>
array=<Array [[], [], [], []] type='4 * var * 2 * timedelta64[W]'>
array=<Array ['', "e\U00034a9e'"] type='2 * string'>
array=<Array [[], ['char']] type='2 * var * string'>
array=<Array [[b'\xd7']] type='1 * var * bytes'>
array=<Array [[[], []], [[]], [], []] type='5 * var * var * 4 * unknown'>
array=<Array [] type='0 * unknown'>
array=<Array [{Rd: []}] type='1 * {Rd: var * datetime64[s]}'>
array=<Array [(''), (..., ...), ..., (..., ...), ('\U0005f041')] type='6 * (string)'>
array=<Array [False] type='1 * union[bytes, bool]'>
array=<Array [36, [b'\x92\xa7\x0b']] type='2 * union[int8, 1 * bytes, unknown]'>
array=<Array [b'5f\x18\xbc', ..., b'5f\x18\xbc'] type='3 * union[string, bytes]'>
array=<Array [0, 0, -5.53e+16] type='3 * union[float32, unknown]'>
array=<Array [??, ??, ??] type='3 * uint32'>
array=<Array [(??, ??)] type='1 * (bytes, union[timedelta64[M], bytes])'>
array=<Array [??, ??, ??, ??, ??] type='5 * var * var * (uint64, bytes)'>
```

The current version generates arrays with `NumpyArray`, `EmptyArray`, string,
and bytestring as leaf contents that can be nested multiple levels deep in
`RegularArray`, `ListOffsetArray`, `ListArray`, `RecordArray`, and
`UnionArray`. Arrays might be virtual, shown as `??` in the output.

### The API of `arrays()`

```python
def arrays(
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    max_size: int = 10,
    allow_nan: bool = False,
    allow_numpy: bool = True,
    allow_empty: bool = True,
    allow_string: bool = True,
    allow_bytestring: bool = True,
    allow_regular: bool = True,
    allow_list_offset: bool = True,
    allow_list: bool = True,
    allow_record: bool = True,
    allow_union: bool = True,
    max_depth: int = 5,
    max_length: int | None = None,
    allow_virtual: bool = True,
):
```

| Parameter | Description |
| --- | --- |
| `dtypes` | A strategy for NumPy scalar dtypes used in `NumpyArray`. If `None`, the default strategy that generates any scalar dtype supported by Awkward Array is used. Does not affect string or bytestring content. |
| `max_size` | Maximum total number of elements in the generated array. Each numerical value counts as one. Each string and bytestring (not character or byte) counts as one. |
| `allow_nan` | No `NaN`/`NaT` values are generated if `False`. |
| `allow_numpy` | No `NumpyArray` is generated if `False`. |
| `allow_empty` | No `EmptyArray` is generated if `False`. |
| `allow_string` | No string content is generated if `False`. Each string (not character) counts toward `max_size`. String layers do not count toward `max_depth`. Unaffected by `dtypes` and `allow_nan`. |
| `allow_bytestring` | No bytestring content is generated if `False`. Each bytestring (not byte) counts toward `max_size`. Bytestring layers do not count toward `max_depth`. Unaffected by `dtypes` and `allow_nan`. |
| `allow_regular` | No `RegularArray` is generated if `False`. |
| `allow_list_offset` | No `ListOffsetArray` is generated if `False`. |
| `allow_list` | No `ListArray` is generated if `False`. |
| `allow_record` | No `RecordArray` is generated if `False`. |
| `allow_union` | No `UnionArray` is generated if `False`. |
| `max_depth` | Maximum nesting depth. Each `RegularArray`, `ListOffsetArray`, `ListArray`, `RecordArray`, and `UnionArray` layer adds one level, excluding those that form string or bytestring content. |
| `max_length` | Maximum `len()` of the generated array. No constraint when `None` (the default). |
| `allow_virtual` | No virtual arrays are generated if `False`. |

## Other strategies

In addition to `arrays()` mentioned above, this package includes other
strategies that generate Awkward Arrays and related data types.

### NumPy

These strategies are related to the section of Awkward Array User Guide ["How to
convert to/from NumPy"][ak-user-guide-numpy].

| Strategy                | Data type                                                      |
| ----------------------- | -------------------------------------------------------------- |
| `from_numpy`            | Awkward Arrays created from NumPy arrays                       |
| `numpy_arrays`          | NumPy arrays that can be converted to Awkward Array            |
| `numpy_dtypes`          | NumPy dtypes (simple or array) supported by Awkward Array      |
| `supported_dtypes`      | NumPy dtypes (simple only) supported by Awkward Array          |
| `supported_dtype_names` | Names of NumPy dtypes (simple only) supported by Awkward Array |

### Python lists

These strategies are related to the section of Awkward Array User Guide ["How to
convert to/from Python objects"][ak-user-guide-python].

| Strategy                   | Data type                                                      |
| -------------------------- | -------------------------------------------------------------- |
| `from_list`                | Awkward Arrays created from Python lists                       |
| `lists`                    | Nested Python lists for which Awkward Arrays can be created    |
| `items_from_dtype`         | Python built-in type values for a given NumPy dtype            |
| `builtin_safe_dtypes`      | NumPy dtypes with corresponding Python built-in types          |
| `builtin_safe_dtype_names` | Names of NumPy dtypes with corresponding Python built-in types |

[ak-user-guide-numpy]: https://awkward-array.org/doc/2.8/user-guide/how-to-convert-numpy.html
[ak-user-guide-python]: https://awkward-array.org/doc/2.8/user-guide/how-to-convert-python.html
