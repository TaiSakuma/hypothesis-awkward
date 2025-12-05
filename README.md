# hypothesis-awkward

_Hypothesis strategies for Awkward Arrays._

[![PyPI - Version](https://img.shields.io/pypi/v/hypothesis-awkward.svg)](https://pypi.org/project/hypothesis-awkward)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hypothesis-awkward.svg)](https://pypi.org/project/hypothesis-awkward)

[![Test Status](https://github.com/TaiSakuma/hypothesis-awkward/actions/workflows/unit-test.yml/badge.svg)](https://github.com/TaiSakuma/hypothesis-awkward/actions/workflows/unit-test.yml)
[![codecov](https://codecov.io/gh/TaiSakuma/hypothesis-awkward/graph/badge.svg?token=cffic9D2b3)](https://codecov.io/gh/TaiSakuma/hypothesis-awkward)

[Hypothesis](https://hypothesis.works/) is a _property-based testing_ library.
Its
[_strategies_](https://hypothesis.readthedocs.io/en/latest/reference/strategies.html)
are Python functions that strategically generate test data that can fail in
_pytest_ or other testing frameworks. Once a test fails, Hypothesis searches for
the simplest sample that causes the same error. Hypothesis automatically
explores edge cases; you do not need to come up with test data manually.

Hypothesis itself includes strategies for
[NumPy](https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#numpy)
and
[pandas](https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#pandas)
data types. [Xarray](https://xarray.dev/) provides [strategies for its data
structure](https://docs.xarray.dev/en/stable/user-guide/testing.html#hypothesis-testing).
The _Apache Arrow_ codebase has [strategies for
PyArrow](https://github.com/apache/arrow/blob/apache-arrow-22.0.0/python/pyarrow/tests/strategies.py),
which are not officially documented in its API reference.

I am putting together Hypothesis strategies that I developed for [Awkward
Array](https://awkward-array.org/) in this package. This is very early work in
progress and still experimental. The APIs may change over time.

## Installation

You can install the package from PyPI using pip:

```bash
pip install hypothesis-awkward
```

This also installs Hypothesis and Awkward Array as dependencies unless they are
already installed.

## A simple example

The strategy `from_numpy` generates Awkward Arrays that are converted from NumPy
arrays. (Internally, it first generates NumPy arrays that can be converted to
Awkward Arrays, then converts them with `ak.from_numpy`.)

The test below converts the generated Awkward Array back to a NumPy array with
`to_numpy` and asserts that the list representations of both arrays are equal.

```python
from hypothesis import given

import awkward as ak
import hypothesis_awkward.strategies as st_ak


@given(ak_array=st_ak.from_numpy(allow_structured=False))
def test_array(ak_array: ak.Array) -> None:
    np_array = ak_array.to_numpy()
    assert ak_array.to_list() == np_array.tolist()
```

## Strategies

So far, I have written strategies based on the first two sections of the
[Awkward Array User
Guide](https://awkward-array.org/doc/2.8/user-guide/index.html): ["How to
convert to/from
NumPy"](https://awkward-array.org/doc/2.8/user-guide/how-to-convert-numpy.html)
and ["How to convert to/from Python
objects"](https://awkward-array.org/doc/2.8/user-guide/how-to-convert-python.html).

### NumPy

These strategies are related to the section ["How to convert to/from
NumPy"](https://awkward-array.org/doc/2.8/user-guide/how-to-convert-numpy.html).

| Strategy                | Data type                                                      |
| ----------------------- | -------------------------------------------------------------- |
| `from_numpy`            | Awkward Arrays created from NumPy arrays                       |
| `numpy_arrays`          | NumPy arrays that can be converted to Awkward Array           |
| `numpy_dtypes`          | NumPy dtypes (simple or array) supported by Awkward Array      |
| `supported_dtypes`      | NumPy dtypes (simple only) supported by Awkward Array          |
| `supported_dtype_names` | Names of NumPy dtypes (simple only) supported by Awkward Array |

### Python lists

These strategies are related to the section ["How to convert to/from Python
objects"](https://awkward-array.org/doc/2.8/user-guide/how-to-convert-python.html).

| Strategy                   | Data type                                                      |
| -------------------------- | -------------------------------------------------------------- |
| `from_list`                | Awkward Arrays created from Python lists                       |
| `lists`                    | Nested Python lists for which Awkward Arrays can be created    |
| `items_from_dtype`         | Python built-in type values for a given NumPy dtype            |
| `builtin_safe_dtypes`      | NumPy dtypes with corresponding Python built-in types          |
| `builtin_safe_dtype_names` | Names of NumPy dtypes with corresponding Python built-in types |

## Perspective

The strategies that I developed so far only generate samples with certain types
of layouts. It is probably possible to build strategies that generate fully
general Awkward Arrays with the [_array
builder_](https://awkward-array.org/doc/2.8/user-guide/how-to-create-arraybuilder.html)
and [_direct
constructors_](https://awkward-array.org/doc/2.8/user-guide/how-to-create-constructors.html),
which would help close in on edge cases in developing tools that use Awkward
Array and even Awkward Array itself.
