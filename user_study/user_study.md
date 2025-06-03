# NBTest User Study

We’d love to get your feedback on [**NBTest**](https://anonymous.4open.science/r/notebook-testing-1805), a tool we built recently.

Jupyter Notebooks are widely used by data scientists and machine learning engineers to develop ML programs. 
However, they often lack proper testing infrastructure, making them difficult to reuse and reproduce. 
As a result, many notebooks are prone to silent regressions—unexpected changes in behavior that go unnoticed over time.

To address this, we built **NBTest**—a regression testing framework designed specifically for Jupyter Notebooks in machine learning workflows. 
NBTest allows the developers to write cell-level
assertions to check the expected intermediate states in the notebook. 
It also provides an assertion generator to automatically generate cell-level assertions for Machine Learning Notebooks.


This short study will guide you through three tasks (which take less than 30 minutes). In return, we’d love to hear what you think. Please fill in the questionaire after you finish all the tasks!
 Your feedback will help us make NBTest better!


## Task 0: Environment Setup

We’ve prepared a Jupyter notebook and a Python environment for you (Recommend using Python 3.9). No GPU required. 

You can choose to build the Python virtual environment through `python venv` or `conda`. 
If you choose `Python venv`, you need to have Python 3.9 installed locally.
If you choose `conda`, conda will install Python 3.9 for you only in the virtual environment.
You could refer to the [conda tutorial](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html) to install conda based on your OS.

Here’s how to get started:

```bash
git clone https://anonymous.4open.science/r/notebook-testing-1805
cd notebook-testing/user_study/nbtest_demo
```

**Method 1: Using conda (recommended)**
```bash
conda env create -f ./environment.yml
conda activate nbtest_env
```

**Method 2: Using python venv**

```sh
python3.9 -m venv nbtest_env 
source ./nbtest_env/bin/activate
pip install -r ./requirements.txt
```

To make sure everything works, try running this:

```bash
jupyter nbconvert --to notebook --inplace --execute ./digits_recognition.ipynb
```

If it runs successfully, you should see cell outputs in the Jupyter Notebook. Also, the terminal output is like this:
```sh
(nbtest_env) (base) usr@laptop nbtest_demo % jupyter nbconvert --to notebook --inplace --execute ./digits_recognition.ipynb      
[NbConvertApp] Converting notebook ./digits_recognition.ipynb to notebook
[NbConvertApp] Writing 45388 bytes to digits_recognition.ipynb
```


## NBTest Assertion API Tutorial

NBTest provides a set of assertion APIs that let users write checks in their notebooks without disrupting their workflow.
Unlike `numpy.testing` or pytest assertions, NBTest assertions are disabled by default, so they don’t interrupt normal notebook execution.
Developers can run and modify their notebooks as usual without triggering assertions.
When they’re ready for testing, they can enable the assertions to check the expected behavior.

In this tutorial, we will show you how to use the nbtest API to write an assertion in Python.
There are two types of assertions APIs:
- General assertions: 
    - `assert_equal(actual, desired)`
    - `assert_allclose(actual, desired, , rtol=1e-07, atol=0)`
    - `assert_true(expr)`
    - `assert_false(expr)`
- Data-related assertions: 
    - `assert_in(item, collection)`
    - `assert_shape(dataframe, expected_shape)`
    - `assert_df_var(dataframe, expected_var, rtol=1e-07, atol=0)`
    - `assert_df_mean(dataframe, expected_mean,rtol=1e-07, atol=0)`
    - `assert_column_types(dataframe, expected_type)`, `assert_column_names(dataframe, expected_col_names)`


**General assertions**: The general assertions, work in the same way as the `numpy.testing` assertion module. For example, `nbtest.assert_equal()` can be used in the same way as in `numpy.testing.assert_equal()`. Below are some usage cases.

*Example usage 1*
```
import pandas as pd
df = pd.read_csv(train_data)
nbtest.assert_equal(df.shape, (1400, 80))
```
In this example, we use `nbtest.assert_equal` to check if the shape of the training data is what we expected.

*Example usage 2*
```
acc = accuracy_score(y_true, y_pred)
nbtest.assert_allclose(acc, 0.98, rtol=1e-3)
```
In this example, we use `nbtest.allclose` to check that the model accuracy is within our expected range. 
Because machine learning involves randomness, we would like to check if the model accuracy lies in a valid range rather than a fixed value.
This is why we used `assert_allclose` rather than `assert_equal`. In this API, `assert_allclose(actual, desired, rtol, atol)`, 
there are two ways to specify the range, i.e., `rtol` and `atol`. 
- `rtol` stands for relative tolerance. `assert_allclose` considers the `actual` and `desired ` are close if `|actual - desired| <= rtol*|desired|`.
- `atol` stands for absolute tolerance. `assert_allclose` considers the `actual` and `desired ` are close if `|actual - desired| <= atol`.


**Data-related Assertions**: For the data-related assertions, most of them take as inputs a pandas dataframe and the expected value of some properties.
- `assert_in(item, collection)`: checks if `item` is in the `collection`.
- `assert_shape(dataframe, expected_shape)`: check the if the shape of `dataframe` is the `expected_shape`.
- `assert_df_var(dataframe, expected_var, rtol=1e-07, atol=0)`: checks if the `variance` of all numerical values in the `dataframe` (`pandas.DataFrame` type) is close to the `expected_var` within the tolerance, which is defined by `rtol` or `atol`. `rtol` and `atol` work in the same as the way in `assert_allclose`.
- `assert_df_mean(dataframe, expected_mean, rtol=1e-07, atol=0)`: checks if the `mean` of all numerical values in the `dataframe` (`pandas.DataFrame` type) is close to the `expected_mean` within the tolerance, which is defined by `rtol` or `atol`.
- `assert_column_types(dataframe, expected_type)`: checks if the types of all columns in the `dataframe` (`pandas.DataFrame` type) are equal to `expected_type` (`List` type).
- `assert_column_names(dataframe, expected_col_names, rtol=1e-07, atol=0)`: checks if the `column names` (`List` type) in the `dataframe` (`pandas.DataFrame` type) are equal to the `expected_col_names`.

*Example usage*
```python
import pandas as pd
df = pd.read_csv(train_data)

nbtest.assert_in("Alley", df.columns) # Check if `age` is a column in the dataframe
nbtest.assert_shape(df, (38, 20))
nbtest.assert_df_var(df, 45.2, rtol=1e-4)
nbtest.assert_df_mean(df, 685.3, rtol=1e-5)
nbtest.assert_column_types(df, ['int64', 'float64', 'object'])
nbtest.assert_column_names(df ['Alley', '2ndFlrSF', '3SsnPorch'])
```
In this example, we used all six data frame assertions to check the shape, variance, mean, column types and column name in the training data.


---

## Task 1: Write assertions manually

In this task, you are asked to write assertions for a copy of the original notebook (i.e., `./task1_digits_recognition.ipynb`). You should write the assertions using the NBTest assertion APIs in the previous tutorial section.
Record the time that you spend on writing all the assertions.
In the end, you could run these assertions to see how good your assertions are.

First, go through the notebook `./task1_digits_recognition.ipynb`. 
This notebook aims to train a simple neural network to classify the digits based on the pixel values.
It first reads from the dataset, does some preprocessing to remove the unnecessary columns,
and defines the neural network. Then it trains the neural network and prints the accuracy in the end.

Tips: You might need to print out the intermediate values to add assertions. 
You can open this notebook with `jupyter-lab ./task1_digits_recognition.ipynb`, and run it cell by cell to inspect the values.



#### Task 1a – Data Integrity Check in "Step 1: Load the dataset and prepare the data"

The code below drops the unnecessary columns in the data frame and then splits them into training and test sets. 
Write an assertion to check that the unnecessary columns have indeed been dropped from `train_df`.
```python
columns_to_drop = ["source", "extra_note", "flag", "id"]
df.drop(columns=columns_to_drop, inplace=True)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
```

#### Task 1b – Model Integrity Check in "Step 2: Define a convolutional neural network"

The code below defines a simple neural network with two convolutional layers.
Write an assertion for `model` to check that the number of layer is 8. 
Hint: `model.layers` is a List.

```python
model = Sequential([
 Conv2D(conv_filters, kernel_size, activation='relu', input_shape=(28, 28, 1)), 
 MaxPooling2D(pool_size=(2, 2)), 
 Conv2D(conv_filters, kernel_size, activation='relu'),
 MaxPooling2D(pool_size=(2, 2)),
 Flatten(), 
 Dense(dense_units, activation='relu'), 
 Dropout(dropout_rate), 
 Dense(10, activation='softmax') 
])
```

#### Task 1c – Model Performance Check in "Step 4: Evaluate the model"

The code below evaluates the model's performance and prints the accuracy.
Write an assertion for `accuracy` to check that the model's accuracy is within the expected range.
We want to check this because if the accuracy is abnormally low or high, it might indicate that something is wrong with the code. 

```python
loss, accuracy = model.evaluate(X_test, y_test, verbose=1) 
```


#### Evaluate these notebooks with assertion 
After you have finished these three assertions, run the command below to see if your assertions pass.

```bash
pytest -v --nbtest ./task1_digits_recognition.ipynb
```

This is a sample output showing that all the tests passed. 
Here, `task1_digits_recognition.ipynb::2_14 PASSED` means that the assertion at Cell 2, Line 2 passed.

```sh
(nbtest_env) usr@laptop nbtest_demo % pytest -v --nbtest ./task1_digits_recognition.ipynb
====================================================== test session starts =======================================================
platform darwin -- Python 3.9.22, pytest-8.3.5, pluggy-1.6.0 -- /usr/miniconda3/envs/nbtest_demo/bin/python3.9
cachedir: .pytest_cache
rootdir: /usr/NBTest
configfile: pyproject.toml
plugins: anyio-4.9.0, nbtest_plugin-0.1.6
collected 3 items                                                                                                                

task1_digits_recognition.ipynb::2_14 PASSED                                                                     [ 33%]
task1_digits_recognition.ipynb::5_15 PASSED                                                                     [ 66%]
task1_digits_recognition.ipynb::17 PASSED                                                                       [100%]

======================================================= 3 passed in 1.61s ========================================================
```

---

## Task 2: Evaluate the assertions automatically generated by NBTest

Instead of writing those assertions manually, NBTest can automatically generate assertions for you. 
NBTest proactively identifies key ML-specific metrics, collects their values across different runs, and automatically generates statistically sound assertions to help detect subtle regressions and inconsistencies.

In this task, we’ll provide you with the assertions that NBTest generated for the notebook above.
You can compare them with the ones you wrote manually to evaluate NBTest usefulness.

We have included the notebook that contains assertions automatically generated by NBTest. 
Open `./task2_digits_recognition_nbtest_assertion.ipynb` and search `nbtest.assert` to see those assertions. 
You can compare these with the ones you wrote manually.
NBTest is particularly useful for capturing randomness. 
For instance, the tolerance value for the `accuracy` variable in the assertion generated by NBTest is quite robust.

Run this command to evaluate the assertions automatically generated by NBTest. You should see that all the assertions pass.

```bash
pytest -v --nbtest ./task2_digits_recognition_nbtest_assertion.ipynb
```

FYI: NBTest can generate a more comprehensive set of assertions.
Developers can review them and selectively add the ones they find useful to their notebook.
To see the full set of generated assertions, check out: `./digits_recognition_nbtest_complete_assertions.ipynb`.


## Task 3: Use NBTest to catch sutble change

Sometimes, changes made to a notebook don’t produce errors or interrupt execution, making them difficult for users to notice.
However, these subtle changes can significantly affect the notebook’s behavior or results.

In this task, you’ll work with a notebook that contains such a subtle change.
First, you’ll run the notebook and see that it executes without any errors.
Then, by enabling NBTest, you’ll see that the automatically generated assertions successfully detect the changes.

In `./task3_digits_recognition_nbtest_assertion.ipynb`, we modified the model definition near the bottom of the notebook by adding **one additional convolutional layer and one max pooling layer**, compared to `./task2_digits_recognition_nbtest_assertion.ipynb`.

Now, run the notebook using the following command:

```bash
jupyter nbconvert --to notebook --inplace --execute ./task3_digits_recognition_nbtest_assertion.ipynb
```

You’ll see that the notebook executes without any errors.
Open it and check the accuracy reported in the last cell — did you notice a *significant drop in accuracy* after adding the two extra layers?

Now, run the notebook with NBTest enabled using `pytest`:

```bash
pytest -v --nbtest ./task3_digits_recognition_nbtest_assertion.ipynb > ./pytest_output.txt
```

Open the `./pytest_output.txt` to check the outputs, you will see something like this.

```
============================= test session starts ==============================
platform darwin -- Python 3.9.22, pytest-8.3.5, pluggy-1.6.0 -- /usr/miniconda3/envs/nbtest_env/bin/python
cachedir: .pytest_cache
rootdir: /usr/NBTest
configfile: pyproject.toml
plugins: nbtest_plugin-0.1.7, anyio-4.7.0
collecting ... collected 5 items

task3_digits_recognition_nbtest_assertion.ipynb::7 PASSED                [ 20%]
task3_digits_recognition_nbtest_assertion.ipynb::6 PASSED                [ 40%]
task3_digits_recognition_nbtest_assertion.ipynb::5 PASSED                [ 60%]
task3_digits_recognition_nbtest_assertion.ipynb::15 FAILED               [ 80%]
task3_digits_recognition_nbtest_assertion.ipynb::17 FAILED               [100%]

=================================== FAILURES ===================================
_____ nbtest_demo/task3_digits_recognition_nbtest_assertion.ipynb::Cell 4 ______
...
[0;31mAssertionError[0m: 
Items are not equal:
key='layers'
key='config'

 ACTUAL: 11
 DESIRED: 9

_____ nbtest_demo/task3_digits_recognition_nbtest_assertion.ipynb::Cell 6 ______
[0;31mAssertionError[0m: 
Not equal to tolerance rtol=1e-07, atol=0.0142097

Mismatched elements: 1 / 1 (100%)
Max absolute difference: 0.03469577
Max relative difference: 0.03597932
 x: array(0.92963)
 y: array(0.964325)

=========================== short test summary info ============================
FAILED task3_digits_recognition_nbtest_assertion.ipynb::15
FAILED task3_digits_recognition_nbtest_assertion.ipynb::17
========================= 2 failed, 3 passed in 9.27s ==========================
```

The `pytest` output shows that the last two assertions failed.
These assertions were checking the model's structure and the accuracy of its predictions.

- Cell 4: The assertion fails because it expected the model to have 9 layers, but it found 11. This is expected, since we added two additional layers.
- Cell 6: The accuracy assertion fails because it expected an accuracy of 0.964 (with the tolerance of ±0.014), but the actual accuracy was 0.929, which is outside the expected range.

This shows how NBTest’s automatically generated assertions can effectively catch subtle performance regressions that might otherwise go unnoticed during normal execution.

