import logging
import re
import nbformat as nbf
import tokenize
from io import StringIO
from pathlib import Path
import os
import re
import pandas as pd
import json

def setup_logger(log_file_path, logger_name, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.propagate = False

    return logger

def set_seed_random_state(code):
    # Step 4: Replace 'random_state' argument values with 'random_seed'

    random_state_pattern = r"(random_state\s*=\s*)\d+(?=\s*[),])"
    cleaned_code = re.sub(random_state_pattern, r"\1random_seed", code)

    return cleaned_code


def preprocess_code(code):
    """
    Preprocess the code by:
    1. Removing lines starting with '!', '%', or containing 'pip install'.
    2. Removing redundant os.walk blocks that print file paths.
    3. Removing all comments and blank lines.
    """
    # Step 1: Remove lines with '!', '%', or 'pip install'
    cleaned_lines = [
        line for line in code.splitlines()
        if not (line.strip().startswith('!') or line.strip().startswith('%') or 'pip install' in line)
    ]
    cleaned_code = "\n".join(cleaned_lines)

    # Step 2: Remove os.walk blocks
    os_walk_pattern = (
        r"for\s+[\w\d_]+,\s*_[^:]*in\s+os\.walk\([^)]+\):\s*\n"
        r"(?:\s*#.*\n)*"  # Optional comments
        r"\s*for\s+[\w\d_]+[^:]*:\s*\n"
        r"(?:\s*#.*\n)*"  # Optional comments
        r"\s*print\(.*\)\s*"
    )
    cleaned_code = re.sub(os_walk_pattern, '', cleaned_code, flags=re.MULTILINE)

    # Step 3: Remove comments and blank lines using tokenize
    final_lines = []
    for line in cleaned_code.splitlines():
        stripped_line = line.strip()
        # Exclude blank lines and comments, preserve other lines with original spaces
        if stripped_line and not stripped_line.startswith("#"):
            final_lines.append(line)
    cleaned_code = "\n".join(final_lines)


    # # Step 4: Replace 'random_state' argument values with 'random_seed'

    # random_state_pattern = r"(random_state\s*=\s*)\d+(?=\s*[),])"
    # cleaned_code = re.sub(random_state_pattern, r"\1random_seed", cleaned_code)


    return cleaned_code

def get_notebook_name(notebook_path):
    notebook_stat_name = Path(notebook_path).stem
    notebook_fname = notebook_stat_name.split("_chebyshev")[0]
    return notebook_fname

# /home/yy2282/project/nb_test/results/results/kaggle__chebyshev_30_0.99_May_19_0102040/kaggle__abdallahahmed400__houseing-price/mutation_houseing-price/added_null/mutant_notebooks/houseing-price_mutant_added_null_0.ipynb
def get_unfinished_mutant_idx(mutant_dir, max_mutant_num, mutant_type, notebook_name):
    pattern = rf'^{notebook_name}_mutant_{mutant_type}_(\d+)\.ipynb$'
    finished_ite = []
    unfinished_ite = []

    for f in os.listdir(mutant_dir):
        match = re.match(pattern, f)
        if match:
            finished_ite.append(match.group(1))

    for i in range(max_mutant_num):
        if str(i) not in finished_ite:
            unfinished_ite.append(i)

    return unfinished_ite

def test_get_unfinished_mutant_idx():
    mutant_dir = "/home/yy2282/project/nb_test/results/results/kaggle__chebyshev_30_0.99_May_19_0102040/kaggle__abdallahahmed400__houseing-price/mutation_houseing-price/added_null/mutant_notebooks"
    max_mutant_num = 4
    mutant_type = "added_null"
    notebook_name = "houseing-price"
    unfinished_ite = get_unfinished_mutant_idx(mutant_dir, max_mutant_num, mutant_type, notebook_name)
    print(unfinished_ite)

def extract_submission_tests(notebook_path):
    try:
        with open(notebook_path, "r") as f:
            nb = json.load(f)
    except Exception as e:
        print(f"[skip] Failed to load notebook {notebook_path}: {e}")
        return []

    saved_dfs = set()
    test_ids = []

    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        source = "".join(cell.get('source', []))

        to_csv_matches = re.findall(r"(\w+)\.to_csv\(([^)]*(?:submission|output)[^)]*)\)", source)
        for varname, _ in to_csv_matches:
            saved_dfs.add(varname)

    for var in saved_dfs:
        pattern = re.compile(rf"nbtest\.assert_\w+\(.*?\b{var}\b.*?test_id\s*=\s*['\"](nbtest_id_[\d_]+)['\"]", re.DOTALL)
        for cell in nb.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue
            source = "".join(cell.get('source', []))
            test_ids.extend(pattern.findall(source))

    return test_ids


def extract_sort_tests(notebook_path):
    try:
        with open(notebook_path, "r") as f:
            nb = json.load(f)
    except Exception as e:
        print(f"[skip] Failed to load notebook {notebook_path}: {e}")
        return []

    sort_df_vars = set()
    test_ids = []

    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        source = "".join(cell.get('source', []))

        sort_assign_matches = re.findall(r"(\w+)\s*=\s*models\.sort_values\([^)]*\)", source)
        for varname in sort_assign_matches:
            sort_df_vars.add(varname)

    for var in sort_df_vars:
        pattern = re.compile(rf"nbtest\.assert_\w+\(.*?\b{var}\b.*?test_id\s*=\s*['\"](nbtest_id_[\d_]+)['\"]", re.DOTALL)
        for cell in nb.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue
            source = "".join(cell.get('source', []))
            test_ids.extend(pattern.findall(source))

    return test_ids

def update_assertion_type(notebook_path, assertion_csv):
    submission_test_ids = extract_submission_tests(notebook_path)
    sort_test_ids = extract_sort_tests(notebook_path)

    assert_df = pd.read_csv(assertion_csv)
    assert_df.to_csv(assertion_csv.replace(".csv", "_original.csv"), index=False)

    # Safely replace types
    mask_sub = assert_df['Assertion_id'].isin(submission_test_ids)
    assert_df.loc[mask_sub, 'Assertion_type'] = assert_df.loc[mask_sub, 'Assertion_type'] \
        .apply(lambda x: x.replace("DATASET", "MODEL_PERF") if isinstance(x, str) else x)

    mask_sort = assert_df['Assertion_id'].isin(sort_test_ids)
    assert_df.loc[mask_sort, 'Assertion_type'] = assert_df.loc[mask_sort, 'Assertion_type'] \
        .apply(lambda x: x.replace("MODEL_ARCH", "MODEL_PERF") if isinstance(x, str) else x)

    assert_df.to_csv(assertion_csv, index=False)

def main():

    ntbk = nbf.read("/home/yy2282/project/nb_test/results/results/MAIN__chebyshev_2_0.95/kaggle__brokerus__ml-project-house-prices-eda-and-7-models/ml-project-house-prices-eda-and-7-models.ipynb", nbf.NO_CONVERT)


    for i, cell in enumerate(ntbk.cells):
        if cell.cell_type == "code":
            cleaned_code = preprocess_code(cell.source)
            print(cleaned_code)


if __name__ == "__main__":
    test_get_unfinished_mutant_idx()
