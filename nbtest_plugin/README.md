## nbtest
#### A pytest plugin for testing Jupyter Notebooks

`nbtest` plugin adds the functionality to define tests (assertions) in Jupyter Notebooks for testing key metrics.
These tests are later collected by pytest, when used with the `--nbtest` flag

#### Assertions supported

- assert_equal
- assert_allclose
- assert_array_almost_equal
- assert_array_less
- assert_greater
- assert_greater_equal
- assert_less
- assert_less_equal
- assert_true
- assert_false

#### Usage examples

##### Testing

```py
import nbtest
import math
import numpy as np

nbtest.assert_equal(round(math.pi, 2), 3.14)
nbtest.assert_greater(math.pi, 5)

print(f'PI: {math.pi}')
```

These tests do not report any errors when the notebook is executed

```bash
$ jupyter execute example.ipynb --output=run.ipynb
[NbClientApp] Executing example.ipynb
[NbClientApp] Executing notebook with kernel: 
[NbClientApp] Save executed results to run.ipynb
```

And output of the cell is:
```
PI: 3.141592653589793
```

Now, we execute the tests using pytest
```bash
$ pytest --nbtest -v ./example.ipynb
========================== test session starts ===========================
platform linux -- Python 3.10.12, pytest-7.1.1, pluggy-1.5.0 -- /usr/bin/python3
cachedir: .pytest_cache
plugins: nbtest-0.1.0, anyio-4.4.0
collected 2 items                                                                                                                          

example.ipynb::0::5 PASSED                                            [ 50%]
example.ipynb::0::6 FAILED                                            [100%]

================================ FAILURES ================================
____________________________ example.ipynb::Cell 0 _______________________
Assertion failed
Cell 0: Assertion error

Input:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
Cell In[1], line 5
      3 import numpy as np
      4 nbtest.assert_equal(round(math.pi, 2), 3.14)
----> 5 nbtest.tc.assertGreater(math.pi, 5)
      6 print(f'PI: {math.pi}')

File /usr/lib/python3.10/unittest/case.py:1244, in TestCase.assertGreater(self, a, b, msg)
   1242 if not a > b:
   1243     standardMsg = '%s not greater than %s' % (safe_repr(a), safe_repr(b))
-> 1244     self.fail(self._formatMessage(msg, standardMsg))

File /usr/lib/python3.10/unittest/case.py:675, in TestCase.fail(self, msg)
    673 def fail(self, msg=None):
    674     """Fail immediately, with the given message."""
--> 675     raise self.failureException(msg)

AssertionError: 3.141592653589793 not greater than 5
```

##### Logging values

When the `COLLECT_VARS` environment variable is set, the asserted values are logged in a csv file when the notebook is executed normally, which can be used for further analysis

```bash
$ export COLLECT_VARS=1
$ jupyter execute example.ipynb --output=run.ipynb
[NbClientApp] Executing example.ipynb
[NbClientApp] Executing notebook with kernel: 
[NbClientApp] Save executed results to run.ipynb
.
.
.
# multiple executions
```

log.csv
```csv
3.14,3.141592653589793
3.14,3.141592653589793
3.14,3.141592653589793
3.14,3.141592653589793
.
.
.
```

