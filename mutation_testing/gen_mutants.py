import nbformat
import os
import ast
import sys
import numpy as np
import pandas as pd
import random
import re
import json
from collections import OrderedDict
import argparse
import traceback

proj_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(proj_folder, '../'))

config_path = os.path.join(proj_folder, "../", "config.json")
global CONFIG_MUTANTS
with open(config_path, "r") as config_file:
    CONFIG_MUTANTS = json.load(config_file)["MUTANT_TYPES"]

from utils.utils import setup_logger, get_notebook_name, get_unfinished_mutant_idx

MAX_MUTANT_COUNT = 4
PANDAS_READERS = ["read_csv", "read_excel", "read_json", "read_parquet"]
SKLEARN_LOADERS = ["clear_data_home", "fetch_20newsgroups", "fetch_california_housing", "fetch_openml", "fetch_file", "fetch_lfw_pairs", "fetch_olivetti_faces", "fetch_rcv1", "get_data_home", "load_diabetes", "load_files", "load_linnerud", "load_sample_images", "load_wine", "load_iris"]
SKLEARN_GENERATORS = ["make_biclusters", "make_blobs", "make_checkerboard", "make_circles", "make_classification", "make_friedman1", "make_friedman2", "make_friedman3", "make_gaussian_quantiles", "make_hastie_10_2", "make_low_rank_matrix", "make_moons", "make_multilabel_classification", "make_regression", "make_s_curve", "make_sparse_coded_signal", "make_sparse_spd_matrix", "make_sparse_uncorrelated", "make_spd_matrix", "make_swiss_roll"]

class EvalRemover(ast.NodeTransformer):
    def __init__(self, logger, df):
        super().__init__()
        self.logger = logger
        self.df = df
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'eval':
            self.logger.debug(f"Removed {ast.unparse(node)}")
            self.df['line'] = [node.lineno]
            self.df['details'] = [f'Removed {ast.unparse(node)}']
            return ast.Constant(value=None)
        return self.generic_visit(node)

class ZeroGradRemover(ast.NodeTransformer):
    def __init__(self, logger, df):
        super().__init__()
        self.logger = logger
        self.df = df
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'zero_grad':
            self.logger.debug(f"Removed {ast.unparse(node)}")
            self.df['line'] = [node.lineno]
            self.df['details'] = [f'Removed {ast.unparse(node)}']
            return ast.Constant(value=None)
        return self.generic_visit(node)

SWAP_RULES = {
    # Scaling
    "StandardScaler": "MinMaxScaler",
    "MinMaxScaler": "StandardScaler",
    "RobustScaler": "StandardScaler",
    "Normalizer": "MinMaxScaler",

    # Feature Selection
    "SelectKBest": "SelectPercentile",
    "SelectPercentile": "SelectKBest",
    "VarianceThreshold(0.0)": "VarianceThreshold(0.01)",
    "VarianceThreshold(0.01)": "VarianceThreshold(0.0)",

    # Dimensionality Reduction
    "PCA": "TruncatedSVD",
    "TruncatedSVD": "PCA",
    "FactorAnalysis": "PCA",
    "KernelPCA(kernel='linear')": "KernelPCA(kernel='rbf')",
    "KernelPCA(kernel='rbf')": "KernelPCA(kernel='linear')",

    # Encoding
    "OneHotEncoder": "OrdinalEncoder",
    "OrdinalEncoder": "OneHotEncoder(handle_unknown='ignore')",

    # Train-Test Split Variants
    "StratifiedKFold": "KFold(shuffle=True, random_state=42)",
    "KFold": "StratifiedKFold(shuffle=True, random_state=42)",
    "RepeatedStratifiedKFold": "RepeatedKFold(n_repeats=2, random_state=42)",
    "RepeatedKFold": "RepeatedStratifiedKFold(n_repeats=2, random_state=42)",
}


class MetricReplacer(ast.NodeTransformer):
    def __init__(self, logger, cell_index, existing_imports, missing_imports):
        self.changed = False
        self.logger = logger
        self.cell_index = cell_index
        self.existing_imports = existing_imports
        self.missing_imports = missing_imports
        self.pipeline_context = {}
        self.swaps_performed = []

    def visit_Call(self, node):
        """ Replaces function calls if they match SWAP_RULES with smart parameter adjustments """
        if isinstance(node.func, ast.Name) and node.func.id in SWAP_RULES:
            old_value, new_value = node.func.id, SWAP_RULES[node.func.id]

            # Add parameter adjustments based on the type of transformer
            if new_value == "SelectPercentile":
                node.func.id = new_value
                has_percentile = False
                for kw in node.keywords:
                    if kw.arg == "percentile":
                        has_percentile = True
                if not has_percentile:
                    node.keywords.append(ast.keyword(arg="percentile", value=ast.Constant(value=50)))

            elif new_value.startswith("TruncatedSVD") or new_value.startswith("PCA"):
                has_components = False
                for i,kw in enumerate(node.keywords):
                    if kw.arg == "n_components":
                        has_components = True
                        if new_value.startswith("TruncatedSVD") and isinstance(kw.value, ast.Constant):
                            if isinstance(kw.value.value, float) and 0 < kw.value.value < 1:
                                self.logger.info(f"Cell {self.cell_index}: Skipped PCA -> TruncatedSVD swap due to incompatible n_components={kw.value.value}")
                                return self.generic_visit(node)
                node.func.id = new_value
                if not has_components:
                    node.keywords.append(ast.keyword(arg="n_components", value=ast.Constant(value=3)))

            elif new_value.startswith("SelectKBest"):
                node.func.id = new_value
                has_k = False
                for kw in node.keywords:
                    if kw.arg == "k":
                        has_k = True
                if not has_k:
                    node.keywords.append(ast.keyword(arg="k", value=ast.Constant(value=3)))

            node.func.id = new_value
            self.logger.info(f"Cell {self.cell_index}: Swapped {old_value} -> {new_value} with parameter adjustments")
            self.changed = True
            if new_value not in self.existing_imports:
                self.missing_imports.add(new_value)

            params = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}

            self.swaps_performed.append({
                'line': getattr(node, 'lineno', 0),
                'old_value': old_value,
                'new_value': new_value,
                'params': params,
            })

        # Handle method calls
        elif isinstance(node.func, ast.Attribute) and node.func.attr in SWAP_RULES:
            old_value, new_value = node.func.attr, SWAP_RULES[node.func.attr]
            node.func.attr = new_value
            self.logger.info(f"Cell {self.cell_index}: Swapped {old_value} -> {new_value}")
            self.changed = True
            if new_value not in self.existing_imports:
                self.missing_imports.add(new_value)

            self.swaps_performed.append({
                'line': getattr(node, 'lineno', 0),
                'old_value': old_value,
                'new_value': new_value,
                'params': {},
            })

        return self.generic_visit(node)


def extract_existing_imports(notebook):
    """ Extract all imported names from the notebook """
    existing_imports = set()
    for cell in notebook.cells:
        if cell.cell_type == "code":
            try:
                tree = ast.parse(cell.source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            existing_imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            existing_imports.add(alias.name)
            except SyntaxError:
                continue
    return existing_imports

CLASS_TO_MODULE = {
    # Preprocessing
    "StandardScaler": "sklearn.preprocessing",
    "MinMaxScaler": "sklearn.preprocessing",
    "RobustScaler": "sklearn.preprocessing",
    "Normalizer": "sklearn.preprocessing",
    "OneHotEncoder": "sklearn.preprocessing",
    "OrdinalEncoder": "sklearn.preprocessing",
    "LabelEncoder": "sklearn.preprocessing",

    # Feature Selection
    "SelectKBest": "sklearn.feature_selection",
    "SelectPercentile": "sklearn.feature_selection",
    "RFE": "sklearn.feature_selection",
    "SelectFromModel": "sklearn.feature_selection",
    "VarianceThreshold": "sklearn.feature_selection",

    # Decomposition
    "PCA": "sklearn.decomposition",
    "TruncatedSVD": "sklearn.decomposition",
    "FactorAnalysis": "sklearn.decomposition",
    "KernelPCA": "sklearn.decomposition",

    # Model Selection
    "KFold": "sklearn.model_selection",
    "StratifiedKFold": "sklearn.model_selection",
    "RepeatedKFold": "sklearn.model_selection",
    "RepeatedStratifiedKFold": "sklearn.model_selection",

    # Other
    "TargetEncoder": "category_encoders"
}

def add_missing_imports(notebook, missing_imports):
    """ Adds missing imports at the top of the notebook, grouped by module """
    if not missing_imports:
        return notebook

    # Group imports by module
    imports_by_module = {}
    for missing in missing_imports:
        if '(' in missing:
            class_name = missing.split('(')[0]
        else:
            class_name = missing

        module = CLASS_TO_MODULE.get(class_name, "sklearn.preprocessing")
        if module not in imports_by_module:
            imports_by_module[module] = []

        imports_by_module[module].append(missing)

    import_statements = []
    for module, classes in imports_by_module.items():
        classes_str = ", ".join(sorted(set(classes)))
        import_statements.append(f"from {module} import {classes_str}")

    import_code = "\n".join(import_statements) + "\n\n"

    if notebook.cells and notebook.cells[0].cell_type == "code":
        notebook.cells[0].source = import_code + notebook.cells[0].source
    else:
        notebook.cells.insert(0, nbformat.v4.new_code_cell(import_code))

    return notebook


def check_chain(call_node: ast.Call):
    methods = []
    current = call_node

    while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        if current.func.attr in ['read_csv', 'read_excel', 'read_json', 'read_parquet', 'DataFrame']:
            return current.func.attr, current
        current = current.func.value

    # Add the base method/function if it exists
    if isinstance(current, ast.Attribute) and hasattr(current, 'attr') and current.attr in ['read_csv', 'read_excel', 'read_json', 'read_parquet', 'DataFrame']:
        return current.attr, current

    # Reverse to get the correct order
    return None, None

def detect_dataframe_sources(notebook, logger):
    """
    Ways DataFrames are created in the notebook:
    - read_csv and other pandas readers
    - Direct DataFrame creation
    """
    df_sources = {}

    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue

        tree = ast.parse(cell.source)

        # Track variable assignments
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                target_names = []
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        target_names.append(target.id)
                    elif isinstance(target, ast.Tuple):
                        for sub_target in target.elts:
                            if isinstance(sub_target, ast.Name):
                                target_names.append(sub_target.id)

                # Check if the right side is a pandas reader
                # DataFrame creation from constructor
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                    if node.value.func.id == 'DataFrame':
                        for name in target_names:
                            df_sources[name] = {
                                'cell_idx': idx,
                                'node': node,
                                'type': 'direct_creation',
                                'creation_line': node.lineno
                            }

                    elif node.value.func.id in SKLEARN_LOADERS:
                        # print(ast.unparse(node))
                        is_frame = False
                        for kw in node.value.keywords:
                            if kw.arg == "as_frame" and isinstance(kw.value, ast.Constant):
                                is_frame = kw.value.value

                        # print(is_frame, target_names)
                        if len(target_names) == 2:
                            df_sources[target_names[0]] = {
                                'cell_idx': idx,
                                'node': node,
                                'type': 'sklearn_creation',
                                'creation_line': node.lineno,
                                'label': target_names[1],
                                'is_frame': is_frame
                            }
                        elif len(target_names) == 1:
                            df_sources[target_names[0]] = {
                                'cell_idx': idx,
                                'node': node,
                                'type': 'sklearn_creation',
                                'creation_line': node.lineno,
                                'is_frame': is_frame
                            }

                elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                    if node.value.func.attr in PANDAS_READERS and isinstance(node.value.args[0], ast.Constant):
                        file_path = node.value.args[0].value
                        for name in target_names:
                            df_sources[name] = {
                                'cell_idx': idx,
                                'node': node,
                                'type': 'pandas_reader',
                                'reader': node.value.func.attr,
                                'creation_line': node.lineno,
                                'file_path': file_path
                            }

                    elif node.value.func.attr == 'DataFrame':
                        for name in target_names:
                            df_sources[name] = {
                                'cell_idx': idx,
                                'node': node,
                                'type': 'direct_creation',
                                'creation_line': node.lineno
                            }

                    elif node.value.func.attr in SKLEARN_LOADERS:
                        # print(ast.unparse(node))
                        is_frame = False
                        for kw in node.value.keywords:
                            if kw.arg == "as_frame" and isinstance(kw.value, ast.Constant):
                                is_frame = kw.value.value

                        # print(is_frame, target_names)
                        if len(target_names) == 2:
                            df_sources[target_names[0]] = {
                                'cell_idx': idx,
                                'node': node,
                                'type': 'sklearn_creation',
                                'creation_line': node.lineno,
                                'label': target_names[1],
                                'is_frame': is_frame
                            }
                        elif len(target_names) == 1:
                            df_sources[target_names[0]] = {
                                'cell_idx': idx,
                                'node': node,
                                'type': 'sklearn_creation',
                                'creation_line': node.lineno,
                                'is_frame': is_frame
                            }

                    else:
                        method, node = check_chain(node.value)

                        # Check if any part of the chain is a pandas reader
                        if method in PANDAS_READERS and isinstance(node.args[0], ast.Constant):
                            for name in target_names:
                                df_sources[name] = {
                                    'cell_idx': idx,
                                    'node': node,
                                    'type': 'pandas_reader',
                                    'reader': method,
                                    'creation_line': node.lineno,
                                    'file_path': node.args[0].value
                                }

                        elif method == 'DataFrame':
                            for name in target_names:
                                df_sources[name] = {
                                    'cell_idx': idx,
                                    'node': node,
                                    'type': 'direct_creation',
                                    'creation_line': node.lineno,
                                }

    logger.debug(f"Detected {len(df_sources)} potential DataFrame sources in the notebook")
    return df_sources

def load_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def save_notebook(notebook, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
    # if os.path.exists(output_path):
    #     logger.debug(f"Exist: {output_path}")
    # else:
    #     logger.debug(f"Nonexist: {output_path}")

def get_inc_fname(original_path, suffix):
    directory = os.path.dirname(original_path)
    if directory == '':
        directory = './'
    base_name, ext = os.path.splitext(os.path.basename(original_path))
    pattern = re.compile(rf"^{re.escape(base_name + suffix)}_(\d+){re.escape(ext)}$")

    # Get all files in the directory matching the pattern
    existing_files = [f for f in os.listdir(directory) if pattern.match(f)]

    if existing_files:
        # Extract existing indices
        existing_ids = [int(pattern.match(f).group(1)) for f in existing_files]
        new_id = max(existing_ids) + 1  # Increment the highest existing ID
    else:
        new_id = 0  # Start from 0 if no file exists

    new_filename = f"{base_name}{suffix}_{new_id}{ext}"
    return os.path.join(directory, new_filename)

# Randomly select some (0.2 by default) numerical numbers and multiply them with a random factor between 10 and 15
def introduce_outliers(notebook, logger, res_dir, mutant_num, outlier_ratio=0.2):
    logger.debug("######### OUTLIER_LOGS #########")
    changed = False
    df_srcs = detect_dataframe_sources(notebook, logger)

    if len(df_srcs) == 0:
        return (notebook, changed)

    # df_srcs_keys = list(df_srcs.keys())[0:4]

    keys = list(df_srcs.keys())
    if len(keys) < mutant_num:
        df_srcs_keys = keys
    else:
        df_srcs_keys = random.sample(keys, mutant_num)


    for df_name in df_srcs_keys:
        changed = False
        src = df_srcs[df_name]

        new_notebook = nbformat.from_dict(notebook)

        logger.debug(f"Dataset found: {df_name}, {str(src)}")
        cell_idx = src['cell_idx']
        cell = new_notebook.cells[cell_idx]

        tree = ast.parse(cell.source)

        if src['type'] == 'pandas_reader' and src['file_path'] is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if (node.func.attr == src['reader'] and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == src['file_path']):

                        try:
                            df = eval(f"pd.{src['reader']}('{src['file_path']}')")
                        except Exception as e:
                            logger.debug(f"Could not open source dataset, due to following:")
                            logger.debug(f"{str(e)}")
                            yield (notebook, changed)
                            break

                        logger.debug(f"Modifying {node.args[0].value} to add some outliers...")

                        numerical_cols = df.select_dtypes(include=[np.floating, np.number]).columns

                        if len(numerical_cols) == 0:
                            logger.debug("No numerical column is found. Skip this cell.")
                            break

                        num_outliers = int(len(df) * outlier_ratio)

                        outlier_indices = np.random.choice(df.index, num_outliers, replace=False)

                        for col in numerical_cols:
                            temp = df.loc[outlier_indices, col] * np.random.uniform(10, 15)
                            df.loc[outlier_indices, col] = temp.astype(df[col].dtype)

                        mod_path = get_inc_fname(src['file_path'], '_with_outliers')
                        df.to_csv(mod_path, index=False)

                        mutant_df_log = pd.DataFrame()
                        mutant_df_log['cell'] = [cell_idx]
                        mutant_df_log['line'] = [node.lineno]
                        mutant_df_log['details'] = [f'outlier_ratio: {outlier_ratio}, Modified indices: {str(outlier_indices)}']
                        mutant_df_log['code'] = [str(CONFIG_MUTANTS['outliers'])]

                        mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

                        mutant_df_path = os.path.join(res_dir, 'mutation_performed.csv')
                        if os.path.exists(mutant_df_path):
                            prev_df = pd.read_csv(mutant_df_path)
                            new_df = pd.concat([prev_df, mutant_df_log], ignore_index=True)
                            new_df.to_csv(mutant_df_path, index=False)
                        else:
                            mutant_df_log.to_csv(mutant_df_path, index=False)

                        node.args[0].value = str(mod_path)
                        changed = True
                        break

            cell.source = ast.unparse(tree)

        elif src['type'] == 'direct_creation':
            creation_line = src['creation_line']
            mutation_code = f"""
import numpy as np
numerical_cols = {df_name}.select_dtypes(include=[np.floating, np.integer]).columns
if len(numerical_cols) > 0:
    num_outliers = int(len({df_name}) * {outlier_ratio})
    outlier_indices = np.random.choice({df_name}.index, num_outliers, replace=False)
    for col in numerical_cols:
        temp = {df_name}.loc[outlier_indices, col] * np.random.uniform(10, 15)
        {df_name}.loc[outlier_indices, col] = temp.astype({df_name}[col].dtype)
"""
            mutation_ast = ast.parse(mutation_code)

            # Insert the mutation code after the DataFrame creation
            new_body = []
            for stmt in tree.body:
                new_body.append(stmt)
                if hasattr(stmt, 'lineno') and stmt.lineno == creation_line:
                    # Add mutation code
                    new_body.extend(mutation_ast.body)

                    # Log mutation details
                    mutant_df_log = pd.DataFrame()
                    mutant_df_log['cell'] = [cell_idx]
                    mutant_df_log['line'] = [creation_line]
                    mutant_df_log['details'] = [f"df_name: {df_name}, type: outliers, outlier_ratio: {outlier_ratio}"]
                    mutant_df_log['code'] = [f"{CONFIG_MUTANTS['outliers']}"]

                    mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

                    mutant_df_path = os.path.join(res_dir, 'mutation_performed.csv')
                    if os.path.exists(mutant_df_path):
                        prev_df = pd.read_csv(mutant_df_path)
                        new_df = pd.concat([prev_df, mutant_df_log], ignore_index=True)
                        new_df.to_csv(mutant_df_path, index=False)
                    else:
                        mutant_df_log.to_csv(mutant_df_path, index=False)

                    changed = True

            tree.body = new_body

            cell.source = ast.unparse(tree)

        elif src['type'] == 'sklearn_creation':
            creation_line = src['creation_line']
            if "label" in src.keys():
                # print("OUTLIERS: ", src["is_frame"])
                if src["is_frame"]:
                    mutation_code = f"""
import numpy as np
numerical_cols = {df_name}.select_dtypes(include=[np.floating, np.integer]).columns
if len(numerical_cols) > 0:
    num_outliers = int(len({df_name}) * {outlier_ratio})
    outlier_indices = np.random.choice({df_name}.index, num_outliers, replace=False)
    for col in numerical_cols:
        temp = {df_name}.loc[outlier_indices, col] * np.random.uniform(10, 15)
        {df_name}.loc[outlier_indices, col] = temp.astype({df_name}[col].dtype)
"""
                else:
                    mutation_code = f"""
import numpy as np
num_samples, num_features = {df_name}.shape

num_outliers = int(num_samples * {outlier_ratio})

if num_outliers > 0:
    outlier_indices = np.random.choice(num_samples, num_outliers, replace=False)
    scale_factors = np.random.uniform(10, 15, size=(num_outliers, num_features))

    {df_name}[outlier_indices] *= scale_factors
"""
            else:
                if src["is_frame"]:
                    mutation_code = f"""
import numpy as np
numerical_cols = {df_name}.data.select_dtypes(include=[np.floating, np.integer]).columns
if len(numerical_cols) > 0:
    num_outliers = int(len({df_name}.data) * {outlier_ratio})
    outlier_indices = np.random.choice({df_name}.data.index, num_outliers, replace=False)
    for col in numerical_cols:
        temp = {df_name}.data.loc[outlier_indices, col] * np.random.uniform(10, 15)
        {df_name}.data.loc[outlier_indices, col] = temp.astype({df_name}.data[col].dtype)
"""
                else:
                    mutation_code = f"""
import numpy as np
num_samples, num_features = {df_name}.data.shape
num_outliers = int(num_samples * {outlier_ratio})

if num_outliers > 0:
    outlier_indices = np.random.choice(num_samples, num_outliers, replace=False)
    scale_factors = np.random.uniform(10, 15, size=(num_outliers, num_features))

    {df_name}.data[outlier_indices] *= scale_factors
"""
            mutation_ast = ast.parse(mutation_code)

            # Insert the mutation code after the DataFrame creation
            new_body = []
            for stmt in tree.body:
                new_body.append(stmt)
                if hasattr(stmt, 'lineno') and stmt.lineno == creation_line:
                    # Add mutation code
                    new_body.extend(mutation_ast.body)

                    # Log mutation details
                    mutant_df_log = pd.DataFrame()
                    mutant_df_log['cell'] = [cell_idx]
                    mutant_df_log['line'] = [creation_line]
                    mutant_df_log['details'] = [f"df_name: {df_name}, type: outliers, outlier_ratio: {outlier_ratio}"]
                    mutant_df_log['code'] = [f"{CONFIG_MUTANTS['outliers']}"]

                    mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

                    mutant_df_path = os.path.join(res_dir, 'mutation_performed.csv')
                    if os.path.exists(mutant_df_path):
                        prev_df = pd.read_csv(mutant_df_path)
                        new_df = pd.concat([prev_df, mutant_df_log], ignore_index=True)
                        new_df.to_csv(mutant_df_path, index=False)
                    else:
                        mutant_df_log.to_csv(mutant_df_path, index=False)

                    changed = True

            tree.body = new_body

            cell.source = ast.unparse(tree)

        yield (new_notebook, changed)

def remove_and_repeat(notebook, logger, res_dir, mutant_num ,removal_ratio=0.3):
    logger.debug("######### REPETITION_LOGS #########")
    changed = False
    df_srcs = detect_dataframe_sources(notebook, logger)

    if len(df_srcs) == 0:
        return (notebook, changed)

    # df_srcs_keys = list(df_srcs.keys())[0:4]
    # df_srcs_keys = random.sample(list(df_srcs.keys()), mutant_num)
    keys = list(df_srcs.keys())
    if len(keys) < mutant_num:
        df_srcs_keys = keys
    else:
        df_srcs_keys = random.sample(keys, mutant_num)

    for df_name in df_srcs_keys:
        changed = False
        src = df_srcs[df_name]

        new_notebook = nbformat.from_dict(notebook)

        logger.debug(f"Dataset found: {df_name}, {str(src)}")
        cell_idx = src['cell_idx']
        cell = new_notebook.cells[cell_idx]

        tree = ast.parse(cell.source)

        if src['type'] == 'pandas_reader' and src['file_path'] is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if (node.func.attr == src['reader'] and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == src['file_path']):

                        try:
                            df = eval(f"pd.{src['reader']}('{src['file_path']}')")
                        except Exception as e:
                            logger.debug(f"Could not open source dataset, due to following:")
                            logger.debug(f"{str(e)}")
                            yield (notebook, changed)
                            break

                        logger.debug(f"Modifying {node.args[0].value} to add some outliers...")

                        num_removed = int(len(df) * removal_ratio)
                        remove_indices = np.random.choice(df.index, size=num_removed, replace=False)
                        replace_indices = np.random.choice(df.index.difference(remove_indices), size=num_removed, replace=False)

                        df.iloc[remove_indices] = df.iloc[replace_indices].values

                        mod_path = get_inc_fname(node.args[0].value, '_with_repetition')

                        new_fname = str(mod_path)

                        df.to_csv(mod_path, index=False)

                        mutant_df_log = pd.DataFrame()
                        mutant_df_log['cell'] = [cell_idx]
                        mutant_df_log['line'] = [node.lineno]
                        mutant_df_log['details'] = [f'removal_ratio: {removal_ratio}, Removed indices: {str(remove_indices)}, Replaced indices: {str(replace_indices)}']
                        mutant_df_log['code'] = [str(CONFIG_MUTANTS["repetition"])]
                        mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

                        mutant_df_path = os.path.join(res_dir, 'mutation_performed.csv')
                        if os.path.exists(mutant_df_path):
                            prev_df = pd.read_csv(mutant_df_path)
                            new_df = pd.concat([prev_df, mutant_df_log], ignore_index=True)
                            new_df.to_csv(mutant_df_path, index=False)
                        else:
                            mutant_df_log.to_csv(mutant_df_path, index=False)

                        node.args[0].value = new_fname
                        changed = True
                        break

            cell.source = ast.unparse(tree)

        elif src['type'] == 'direct_creation':
            creation_line = src['creation_line']
            mutation_code = f"""
import numpy as np
num_removed = int(len({df_name}) * {removal_ratio})
remove_indices = np.random.choice({df_name}.index, size=num_removed, replace=False)
replace_indices = np.random.choice({df_name}.index.difference(remove_indices), size=num_removed, replace=False)

{df_name}.iloc[remove_indices] = {df_name}.iloc[replace_indices].values
"""
            mutation_ast = ast.parse(mutation_code)

            # Insert the mutation code after the DataFrame creation
            new_body = []
            for stmt in tree.body:
                new_body.append(stmt)
                if hasattr(stmt, 'lineno') and stmt.lineno == creation_line:
                    # Add mutation code
                    new_body.extend(mutation_ast.body)

                    # Log mutation details
                    mutant_df_log = pd.DataFrame()
                    mutant_df_log['cell'] = [cell_idx]
                    mutant_df_log['line'] = [creation_line]
                    mutant_df_log['details'] = [f'removal_ratio: {removal_ratio}']
                    mutant_df_log['code'] = [str(CONFIG_MUTANTS["repetition"])]

                    mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

                    mutant_df_path = os.path.join(res_dir, 'mutation_performed.csv')
                    if os.path.exists(mutant_df_path):
                        prev_df = pd.read_csv(mutant_df_path)
                        new_df = pd.concat([prev_df, mutant_df_log], ignore_index=True)
                        new_df.to_csv(mutant_df_path, index=False)
                    else:
                        mutant_df_log.to_csv(mutant_df_path, index=False)
                    changed = True

            tree.body = new_body

            cell.source = ast.unparse(tree)

        elif src['type'] == 'sklearn_creation':
            creation_line = src['creation_line']
            if "label" in src.keys():
                if src["is_frame"]:
                    mutation_code = f"""
import numpy as np
num_removed = int(len({df_name}) * {removal_ratio})
remove_indices = np.random.choice({df_name}.index, size=num_removed, replace=False)
replace_indices = np.random.choice({df_name}.index.difference(remove_indices), size=num_removed, replace=False)

{df_name}.iloc[remove_indices] = {df_name}.iloc[replace_indices].values
"""
                else:
                    mutation_code = f"""
num_samples = {df_name}.shape[0]

num_removed = int(num_samples * {removal_ratio})
if num_removed != 0:
    remove_indices = np.random.choice(num_samples, size=num_removed, replace=False)

    available_indices = np.setdiff1d(np.arange(num_samples), remove_indices)
    replace_indices = np.random.choice(available_indices, size=num_removed, replace=True)

    {df_name}[remove_indices] = {df_name}[replace_indices]
"""
            else:
                if src["is_frame"]:
                    mutation_code = f"""
import numpy as np
num_removed = int(len({df_name}.data) * {removal_ratio})
remove_indices = np.random.choice({df_name}.data.index, size=num_removed, replace=False)
replace_indices = np.random.choice({df_name}.data.index.difference(remove_indices), size=num_removed, replace=False)

{df_name}.data.iloc[remove_indices] = {df_name}.data.iloc[replace_indices].values
{df_name}.target.iloc[remove_indeces] = {df_name}.target.iloc[replace_indices].values
"""

                else:
                    mutation_code = f"""
import numpy as np
num_samples = {df_name}.data.shape[0]

num_removed = int(num_samples * {removal_ratio})
if num_removed != 0:
    remove_indices = np.random.choice(num_samples, size=num_removed, replace=False)

    available_indices = np.setdiff1d(np.arange(num_samples), remove_indices)
    replace_indices = np.random.choice(available_indices, size=num_removed, replace=True)

    {df_name}.data[remove_indices] = {df_name}.data[replace_indices]
    {df_name}.target[remove_indices] = {df_name}.data[replace_indices]
"""

            mutation_ast = ast.parse(mutation_code)

            # Insert the mutation code after the DataFrame creation
            new_body = []
            for stmt in tree.body:
                new_body.append(stmt)
                if hasattr(stmt, 'lineno') and stmt.lineno == creation_line:
                    # Add mutation code
                    new_body.extend(mutation_ast.body)

                    # Log mutation details
                    mutant_df_log = pd.DataFrame()
                    mutant_df_log['cell'] = [cell_idx]
                    mutant_df_log['line'] = [creation_line]
                    mutant_df_log['details'] = [f'removal_ratio: {removal_ratio}']
                    mutant_df_log['code'] = [str(CONFIG_MUTANTS["repetition"])]

                    mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

                    mutant_df_path = os.path.join(res_dir, 'mutation_performed.csv')
                    if os.path.exists(mutant_df_path):
                        prev_df = pd.read_csv(mutant_df_path)
                        new_df = pd.concat([prev_df, mutant_df_log], ignore_index=True)
                        new_df.to_csv(mutant_df_path, index=False)
                    else:
                        mutant_df_log.to_csv(mutant_df_path, index=False)
                    changed = True

            tree.body = new_body

            cell.source = ast.unparse(tree)

        yield (new_notebook, changed)

def add_null(notebook, logger, res_dir, mutant_num, rows_ratio=0.1, subset_ratio=0.1):
    logger.debug("######### NULL_LOGS #########")
    changed = False
    df_srcs = detect_dataframe_sources(notebook, logger)

    if len(df_srcs) == 0:
        return (notebook, changed)

    # df_srcs_keys = list(df_srcs.keys())[0:4]
    # df_srcs_keys = random.sample(list(df_srcs.keys()), mutant_num)
    keys = list(df_srcs.keys())
    if len(keys) < mutant_num:
        df_srcs_keys = keys
    else:
        df_srcs_keys = random.sample(keys, mutant_num)

    for df_name in df_srcs_keys:
        changed = False
        src = df_srcs[df_name]

        new_notebook = nbformat.from_dict(notebook)

        logger.debug(f"Dataset found: {df_name}, {str(src)}")
        cell_idx = src['cell_idx']
        cell = new_notebook.cells[cell_idx]

        tree = ast.parse(cell.source)

        if src['type'] == 'pandas_reader' and src['file_path'] is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if (node.func.attr == src['reader'] and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == src['file_path']):

                        try:
                            df = eval(f"pd.{src['reader']}('{src['file_path']}')")
                        except Exception as e:
                            logger.debug(f"Could not open source dataset, due to following:")
                            logger.debug(f"{str(e)}")
                            yield (notebook, changed)
                            break

                        logger.debug(f"Modifying {node.args[0].value} to add some outliers...")

                        numerical_cols = df.select_dtypes(include=[np.number]).columns

                        if len(numerical_cols) == 0:
                            logger.debug("No numerical column is found. Skip this cell.")
                            break

                        num_cols_to_null = max(1, int(len(numerical_cols) * subset_ratio))
                        cols_to_null = np.random.choice(numerical_cols, size=num_cols_to_null, replace=False)

                        all_null_idxs = []

                        for col in cols_to_null:
                            num_nulls = max(1, int(len(df) * rows_ratio))
                            null_indices = np.random.choice(df.index, size=num_nulls, replace=False)
                            all_null_idxs.extend(null_indices)
                            df.loc[null_indices, col] = None

                        mod_path = get_inc_fname(node.args[0].value, '_with_null')
                        new_fname = str(mod_path)

                        df.to_csv(mod_path, index=False)

                        mutant_df_log = pd.DataFrame()
                        mutant_df_log['cell'] = [cell_idx]
                        mutant_df_log['line'] = [node.lineno]
                        mutant_df_log['details'] = [f'rows_ratio: {rows_ratio}, subset_ratio: {subset_ratio}, Modified indices: {str(all_null_idxs)}, Modified columns: {str(cols_to_null)}']
                        mutant_df_log['code'] = [str(CONFIG_MUTANTS['added_null'])]
                        mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

                        mutant_df_path = os.path.join(res_dir, 'mutation_performed.csv')
                        if os.path.exists(mutant_df_path):
                            prev_df = pd.read_csv(mutant_df_path)
                            new_df = pd.concat([prev_df, mutant_df_log], ignore_index=True)
                            new_df.to_csv(mutant_df_path, index=False)
                        else:
                            mutant_df_log.to_csv(mutant_df_path, index=False)
                        changed = True
                        node.args[0].value = new_fname
                        changed = True

            cell.source = ast.unparse(tree)

        elif src['type'] == 'direct_creation':
            creation_line = src['creation_line']
            mutation_code = f"""
import numpy as np
num_cols_to_null = max(1, int(len({df_name}.columns) * {subset_ratio}))
cols_to_null = np.random.choice({df_name}.columns, size=num_cols_to_null, replace=False)

for col in cols_to_null:
    num_nulls = int(len({df_name}) * {rows_ratio})
    null_indices = np.random.choice({df_name}.index, size=num_nulls, replace=False)
    {df_name}.loc[null_indices, col] = np.nan
"""
            mutation_ast = ast.parse(mutation_code)

            # Insert the mutation code after the DataFrame creation
            new_body = []
            for stmt in tree.body:
                new_body.append(stmt)
                if hasattr(stmt, 'lineno') and stmt.lineno == creation_line:
                    # Add mutation code
                    new_body.extend(mutation_ast.body)

                    # Log mutation details
                    mutant_df_log = pd.DataFrame()
                    mutant_df_log['cell'] = [cell_idx]
                    mutant_df_log['line'] = [creation_line]
                    mutant_df_log['details'] = [f'rows_ratio: {rows_ratio}, subset_ratio: {subset_ratio}']
                    mutant_df_log['code'] = [str(CONFIG_MUTANTS['added_null'])]

                    mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

                    mutant_df_path = os.path.join(res_dir, 'mutation_performed.csv')
                    if os.path.exists(mutant_df_path):
                        prev_df = pd.read_csv(mutant_df_path)
                        new_df = pd.concat([prev_df, mutant_df_log], ignore_index=True)
                        new_df.to_csv(mutant_df_path, index=False)
                    else:
                        mutant_df_log.to_csv(mutant_df_path, index=False)
                    changed = True

            tree.body = new_body

            cell.source = ast.unparse(tree)

        elif src['type'] == 'sklearn_creation':
            creation_line = src['creation_line']
            if "label" in src.keys():
                if src["is_frame"]:
                    mutation_code = f"""
import numpy as np
num_cols_to_null = max(1, int(len({df_name}.columns) * {subset_ratio}))
cols_to_null = np.random.choice({df_name}.columns, size=num_cols_to_null, replace=False)

for col in cols_to_null:
    num_nulls = int(len({df_name}) * {rows_ratio})
    null_indices = np.random.choice({df_name}.index, size=num_nulls, replace=False)
    {df_name}.loc[null_indices, col] = np.nan
"""

                else:
                    mutation_code = f"""
num_samples, num_features = {df_name}.shape

# Select subset of columns to apply NaNs
num_cols_to_null = max(1, int(num_features * {subset_ratio}))  # Ensure at least one column
cols_to_null = np.random.choice(num_features, size=num_cols_to_null, replace=False)

for col in cols_to_null:
    # Select subset of rows in the chosen column
    num_nulls = max(1, int(num_samples * {rows_ratio}))  # Ensure at least one NaN
    null_indices = np.random.choice(num_samples, size=num_nulls, replace=False)

    # Set selected values to NaN
    {df_name}[null_indices, col] = np.nan
"""
            else:
                if src["is_frame"]:
                    mutation_code = f"""
import numpy as np
num_cols_to_null = max(1, int(len({df_name}.data.columns) * {subset_ratio}))
cols_to_null = np.random.choice({df_name}.data.columns, size=num_cols_to_null, replace=False)

{df_name}.data = {df_name}.data.astype(float)

for col in cols_to_null:
    num_nulls = int(len({df_name}.data) * {rows_ratio})
    null_indices = np.random.choice({df_name}.data.index, size=num_nulls, replace=False)
    {df_name}.data.loc[null_indices, col] = np.nan
"""
                else:
                    mutation_code = f"""
import numpy as np
num_samples, num_features = {df_name}.data.shape

# Select subset of columns to apply NaNs
num_cols_to_null = max(1, int(num_features * {subset_ratio}))  # Ensure at least one column
cols_to_null = np.random.choice(num_features, size=num_cols_to_null, replace=False)

{df_name}.data = {df_name}.data.astype(float)

for col in cols_to_null:
    # Select subset of rows in the chosen column
    num_nulls = max(1, int(num_samples * {rows_ratio}))  # Ensure at least one NaN
    null_indices = np.random.choice(num_samples, size=num_nulls, replace=False)

    # Set selected values to NaN
    {df_name}.data[null_indices, col] = np.nan
"""
            mutation_ast = ast.parse(mutation_code)

            # Insert the mutation code after the DataFrame creation
            new_body = []
            for stmt in tree.body:
                new_body.append(stmt)
                if hasattr(stmt, 'lineno') and stmt.lineno == creation_line:
                    # Add mutation code
                    new_body.extend(mutation_ast.body)

                    # Log mutation details
                    mutant_df_log = pd.DataFrame()
                    mutant_df_log['cell'] = [cell_idx]
                    mutant_df_log['line'] = [creation_line]
                    mutant_df_log['details'] = [f'rows_ratio: {rows_ratio}, subset_ratio: {subset_ratio}']
                    mutant_df_log['code'] = [str(CONFIG_MUTANTS['added_null'])]

                    mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

                    mutant_df_path = os.path.join(res_dir, 'mutation_performed.csv')
                    if os.path.exists(mutant_df_path):
                        prev_df = pd.read_csv(mutant_df_path)
                        new_df = pd.concat([prev_df, mutant_df_log], ignore_index=True)
                        new_df.to_csv(mutant_df_path, index=False)
                    else:
                        mutant_df_log.to_csv(mutant_df_path, index=False)
                    changed = True

            tree.body = new_body

            cell.source = ast.unparse(tree)

        yield (new_notebook, changed)

# Removes zero_grad() for torch optimizers
def remove_zero_grad(notebook, logger, res_dir, mutant_num=1):
    logger.debug("######### ZERO_GRAD_LOGS #########")
    changed = False
    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code' and '.zero_grad()' in cell.source:
            tree = ast.parse(cell.source)

            mutant_df_log = pd.DataFrame()
            mutant_df_log['cell'] = [idx]

            tree = ZeroGradRemover(logger, mutant_df_log).visit(tree)
            ast.fix_missing_locations(tree)
            tmp = ast.unparse(tree)

            if (tmp != cell.source):
                changed = True
                cell.source = tmp

            mutant_df_log['code'] = [str(CONFIG_MUTANTS['remove_torch_zero_grad'])]
            mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

            mutant_df_log.to_csv(os.path.join(res_dir, 'mutation_performed.csv'), index=False)

    return [(notebook, changed)]

# Removes eval() mode setting for torch models
def remove_eval(notebook, logger, res_dir, mutant_num=1):
    logger.debug("######### EVAL_LOGS #########")
    changed = False
    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code' and 'eval()' in cell.source:
            tree = ast.parse(cell.source)

            mutant_df_log = pd.DataFrame()
            mutant_df_log['cell'] = [idx]

            tree = EvalRemover(logger, mutant_df_log).visit(tree)
            ast.fix_missing_locations(tree)
            tmp = ast.unparse(tree)

            if (tmp != cell.source):
                changed = True
                cell.source = tmp

            mutant_df_log['code'] = [str(CONFIG_MUTANTS['remove_torch_zero_grad'])]
            mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

            mutant_df_log.to_csv(os.path.join(res_dir, 'mutation_performed.csv'), index=False)

    return [(notebook, changed)]

# Modify hyperparameters for model classes and methods
def modify_hyperparameters(notebook, logger, res_dir, modification_type='modify', modification_ratio=0.5, notebook_id=None):
    logger.debug(f"######### HYPERPARAMETER_{modification_type}_LOGS #########")
    modified = False
    all_mutations = []

    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            original_source = cell.source
            if '!pip' in original_source or '%matplotlib' in original_source:
                continue
            tree = ast.parse(cell.source)

            # Common model classes
            model_classes = [
                # Traditional ML
                'LinearRegression', 'LogisticRegression', 'RandomForestClassifier',
                'RandomForestRegressor', 'SVC', 'SVR', 'GradientBoostingClassifier',
                'GradientBoostingRegressor', 'XGBClassifier', 'XGBRegressor',
                'LGBMClassifier', 'LGBMRegressor', 'DecisionTreeClassifier',
                'KNeighborsClassifier', 'MLPClassifier', 'Pipeline',
                'GridSearchCV', 'RandomizedSearchCV',

                # Deep Learning - Keras/TF
                'Sequential', 'Model', 'Dense', 'Conv1D', 'Conv2D', 'Conv3D',
                'LSTM', 'GRU', 'Bidirectional', 'Dropout', 'BatchNormalization',
                'MaxPooling1D', 'MaxPooling2D', 'MaxPooling3D',
                'AveragePooling1D', 'AveragePooling2D', 'AveragePooling3D',

                # PyTorch
                'Linear', 'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d',
                'ConvTranspose3d', 'MaxPool1d', 'MaxPool2d', 'MaxPool3d', 'AvgPool1d',
                'AvgPool2d', 'AvgPool3d', 'LSTM', 'GRU', 'RNN', 'Transformer',
                'TransformerEncoder', 'TransformerDecoder', 'MultiheadAttention'
            ]

            # Callback classes
            callback_classes = [
                'EarlyStopping', 'ModelCheckpoint', 'ReduceLROnPlateau',
                'TensorBoard', 'CSVLogger', 'LearningRateScheduler'
            ]

            # Common fit/compile methods to look for
            dl_methods = ['fit', 'fit_generator', 'compile', 'to', 'add']

            # Optimizer classes
            optimizer_classes = ['SGD', 'Adam', 'RMSprop', 'Adagrad']

            # Parameter ranges by category
            dl_param_ranges = {
                'units': (16, 1024),  # (min, max)
                'filters': (16, 512),
                'kernel_size': (1, 7),
                'pool_size': (1, 4),
                'strides': (1, 3),
                'rate': (0.1, 0.5),  # for dropout
                'momentum': (0.7, 0.99),  # for batch norm
                'eps': (1e-8, 1e-3),  # for batch norm
                'hidden_size': (16, 1024),  # for RNNs
                'num_layers': (1, 8),  # for RNNs
                'in_features': (16, 1024),  # PyTorch
                'out_features': (16, 1024),  # PyTorch
                'in_channels': (16, 512),  # PyTorch
                'out_channels': (16, 512),  # PyTorch
                # Method parameters
                'epochs': (5, 300),
                'batch_size': (8, 256),
                'learning_rate': (1e-5, 1e-1),
                'lr': (1e-5, 1e-1),
                'weight_decay': (1e-6, 1e-3),
                # Callback parameters
                'patience': (2, 30),
                'min_delta': (1e-5, 1e-2),
                'cooldown': (0, 10),
                'min_lr': (1e-7, 1e-4),
                'factor': (0.1, 0.9),
            }

            # Essential parameters by class
            essential_params = {
                'Dense': ['units'],
                'Linear': ['in_features', 'out_features'],
                'Conv2D': ['filters'],
                'Conv2d': ['in_channels', 'out_channels'],
                'LSTM': ['hidden_size'],
                'Dropout': ['rate'],
                # Method essentials
                'fit': ['X', 'y', 'data', 'labels', 'X_train', 'y_train', 'eval_set', 'eval_names'],
                'compile': ['loss'],
                # Callbacks essentials
                'EarlyStopping': ['monitor'],
                'ModelCheckpoint': ['filepath'],
                'ReduceLROnPlateau': ['monitor']
            }

            # Option choices by parameter
            param_options = {
                'activation': ['relu', 'sigmoid', 'tanh', 'elu', 'selu', 'softmax', 'linear'],
                'optimizer': ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam'],
                'loss': ['categorical_crossentropy', 'binary_crossentropy', 'mse', 'mae',
                        'sparse_categorical_crossentropy', 'kl_divergence'],
                'monitor': ['val_loss', 'val_accuracy', 'loss', 'accuracy', 'val_auc', 'auc',
                           'val_precision', 'precision', 'val_recall', 'recall'],
                'mode': ['auto', 'min', 'max']
            }

            # Helper functions
            def get_node_value(node):
                """Extract value from AST node regardless of node type"""
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool, type(None))):
                    return node.value
                elif isinstance(node, ast.Str):
                    return node.s
                elif isinstance(node, ast.NameConstant):
                    return node.value
                return None

            def set_node_value(node, value):
                """Set value to AST node regardless of node type"""
                if isinstance(node, ast.Num):
                    node.n = value
                elif isinstance(node, ast.Constant):
                    node.value = value
                elif isinstance(node, ast.Str):
                    node.s = value
                elif isinstance(node, ast.NameConstant):
                    node.value = value
                return node

            def modify_numeric_value(current_value, param_name):
                """Modify a numeric parameter value based on its type and name"""
                if isinstance(current_value, int):
                    if param_name == 'epochs':
                        # Randomly increase or decrease epochs
                        if random.choice([True, False]):
                            return min(dl_param_ranges[param_name][1], current_value * random.randint(2, 4))
                        else:
                            return max(dl_param_ranges[param_name][0], current_value // random.randint(2, 3))
                    elif param_name == 'batch_size':
                        # Change batch size by powers of 2
                        batch_sizes = [8, 16, 32, 64, 128, 256]
                        idx = batch_sizes.index(current_value) if current_value in batch_sizes else 0
                        if idx < len(batch_sizes) - 1 and random.choice([True, False]):
                            return batch_sizes[idx + 1]
                        elif idx > 0:
                            return batch_sizes[idx - 1]
                        else:
                            return batch_sizes[1]
                    elif param_name == 'patience':
                        # For patience parameter, increase or decrease significantly
                        if random.choice([True, False]):
                            return max(1, min(dl_param_ranges[param_name][1], current_value * random.randint(2, 3)))
                        else:
                            return max(dl_param_ranges[param_name][0], current_value // random.randint(2, 3))
                    else:
                        # For other integer parameters
                        return max(1, current_value * random.randint(2, 5))

                elif isinstance(current_value, float):
                    if param_name in ['learning_rate', 'lr']:
                        # Shift by order of magnitude
                        if random.choice([True, False]):
                            return min(dl_param_ranges[param_name][1], current_value * 10)
                        else:
                            return max(dl_param_ranges[param_name][0], current_value / 10)
                    elif param_name == 'rate':  # Dropout
                        return random.uniform(0.1, 0.9)
                    elif param_name in ['momentum', 'eps']:
                        min_val, max_val = dl_param_ranges.get(param_name, (0.7, 0.99))
                        return random.uniform(min_val, max_val)
                    elif param_name == 'min_delta':
                        # For min_delta in EarlyStopping
                        return max(dl_param_ranges[param_name][0],
                                  min(dl_param_ranges[param_name][1],
                                      current_value * random.uniform(0.1, 10.0)))
                    elif param_name == 'factor':
                        # For factor in ReduceLROnPlateau
                        return max(dl_param_ranges[param_name][0],
                                  min(dl_param_ranges[param_name][1],
                                      current_value * random.uniform(0.5, 1.5)))
                    else:
                        # Generic float modification
                        return current_value * random.uniform(1.5, 3.0)

            def modify_string_value(current_value, param_name):
                """Modify a string parameter based on parameter name"""
                if param_name in param_options:
                    options = param_options[param_name].copy()
                    if current_value in options:
                        options.remove(current_value)
                    return random.choice(options)
                return current_value

            def modify_beta_tuple(tuple_node):
                """Modify beta parameters for optimizers"""
                if len(tuple_node.elts) != 2:
                    return tuple_node

                for i, elt in enumerate(tuple_node.elts):
                    current = get_node_value(elt)
                    if current is not None:
                        if i == 0:  # First beta
                            new_value = max(0.8, min(0.99, current * random.uniform(0.9, 1.1)))
                        else:  # Second beta
                            new_value = max(0.9, min(0.9999, current * random.uniform(0.99, 1.01)))
                        set_node_value(elt, new_value)

                return tuple_node

            def is_essential_param(param_name, context_name):
                """Check if parameter is essential for a given context"""
                if context_name in essential_params and param_name in essential_params[context_name]:
                    return True
                # Check method params across different method types
                for method_key in ['fit', 'compile']:
                    if method_key in essential_params and param_name in essential_params[method_key]:
                        return True
                return False

            def modify_parameter(keyword, context_name=''):
                """Modify a single parameter"""
                param_name = keyword.arg
                current_value = get_node_value(keyword.value)

                # Special case for beta tuples in optimizers
                if param_name == 'betas' and isinstance(keyword.value, ast.Tuple):
                    keyword.value = modify_beta_tuple(keyword.value)
                    return True

                if current_value is None:
                    return False

                # Handle different value types
                if isinstance(current_value, bool):
                    set_node_value(keyword.value, not current_value)

                elif isinstance(current_value, (int, float)):
                    # Check if we have specific range constraints
                    if param_name in dl_param_ranges:
                        min_val, max_val = dl_param_ranges[param_name]
                        new_value = modify_numeric_value(current_value, param_name)
                        new_value = max(min_val, min(max_val, new_value))
                    else:
                        new_value = modify_numeric_value(current_value, param_name)
                    set_node_value(keyword.value, new_value)

                elif isinstance(current_value, str):
                    new_value = modify_string_value(current_value, param_name)
                    set_node_value(keyword.value, new_value)

                return True

            # Helper function to identify callbacks in lists or variables
            def find_and_modify_callbacks(node, parent_node=None):
                """Find and modify callbacks in a list or as a standalone variable"""
                nonlocal modified
                modified_callbacks = False

                # Handle callback list initialization like callbacks=[EarlyStopping(...), ...]
                if isinstance(node, ast.List):
                    for i, elt in enumerate(node.elts):
                        if isinstance(elt, ast.Call):
                            if handle_callback_node(elt):
                                modified_callbacks = True
                                # Record this mutation for each element in the list
                                if parent_node is not None and hasattr(elt, 'func'):
                                    cb_name = elt.func.id if isinstance(elt.func, ast.Name) else elt.func.attr
                                    cb_params = ", ".join([kw.arg for kw in elt.keywords if isinstance(kw.arg, str)])

                                    mutation_data = {
                                        'notebook_id': notebook_id,
                                        'cell': idx,
                                        'line': getattr(elt, 'lineno', getattr(parent_node, 'lineno', 0)),
                                        'details': f'modification_ratio: {modification_ratio}, Modifying callback "{cb_name}" with parameters: {cb_params}',
                                        'code': str(CONFIG_MUTANTS[f"{modification_type}_hyperparameters"])
                                    }
                                    all_mutations.append(mutation_data)

                # Handle variable assignment to a callback like es = EarlyStopping(...)
                elif isinstance(node, ast.Call):
                    if handle_callback_node(node):
                        modified_callbacks = True

                        # Record this mutation for direct call
                        if parent_node is not None and hasattr(node, 'func'):
                            cb_name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                            cb_params = ", ".join([kw.arg for kw in node.keywords if isinstance(kw.arg, str)])

                            mutation_data = {
                                'notebook_id': notebook_id,
                                'cell': idx,
                                'line': getattr(node, 'lineno', getattr(parent_node, 'lineno', 0)),
                                'details': f'modification_ratio: {modification_ratio}, Modifying callback "{cb_name}" with parameters: {cb_params}',
                                'code': str(CONFIG_MUTANTS[f"{modification_type}_hyperparameters"])
                            }
                            all_mutations.append(mutation_data)

                return modified_callbacks

            def handle_callback_node(node):
                """Process identified callback node"""
                if not isinstance(node, ast.Call) or not hasattr(node, 'func') or not node.keywords:
                    return False

                callback_name = None

                # Check for callback class name
                if isinstance(node.func, ast.Name) and node.func.id in callback_classes:
                    callback_name = node.func.id
                elif isinstance(node.func, ast.Attribute) and node.func.attr in callback_classes:
                    callback_name = node.func.attr

                if not callback_name:
                    return False

                # Now we found a callback, let's modify or remove parameters
                if len(node.keywords) <= 8:
                    num_to_modify = max(1, int(len(node.keywords) * modification_ratio))
                else:
                    num_to_modify = 4

                params_to_modify = random.sample(node.keywords, num_to_modify)

                params = ", ".join([keyword.arg for keyword in params_to_modify if isinstance(keyword.arg, str)])

                if modification_type == 'remove':
                    logger.debug(f"Removing callback '{callback_name}' with parameters: {params}")
                else:
                    logger.debug(f"Modifying callback '{callback_name}' with parameters: {params}")

                # Process selected parameters
                local_modified = False
                for keyword in params_to_modify:
                    if modification_type == 'remove':
                        # Skip essential parameters
                        if is_essential_param(keyword.arg, callback_name):
                            continue
                        node.keywords.remove(keyword)
                        local_modified = True
                    else:  # modify
                        if modify_parameter(keyword, callback_name):
                            local_modified = True

                return local_modified

            # Function to find Fit method callbacks parameter
            def handle_fit_callbacks(node):
                """Find and process callbacks in model.fit(..., callbacks=[...]) calls"""
                nonlocal modified

                if not isinstance(node, ast.Call) or not hasattr(node, 'func'):
                    return

                # Check if this is a fit method call
                is_fit_call = False
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'fit':
                    is_fit_call = True

                if not is_fit_call:
                    return

                # Look for callbacks parameter
                for keyword in node.keywords:
                    if keyword.arg == 'callbacks' and isinstance(keyword.value, ast.List):
                        if find_and_modify_callbacks(keyword.value, node):
                            modified = True

            # Main AST traversal
            for node in ast.walk(tree):
                # First, check regular model class, method and optimizer calls
                if isinstance(node, ast.Call) and hasattr(node, 'func') and node.keywords:
                    # Identify call type
                    call_type = None
                    call_name = None

                    # Model class initialization
                    if isinstance(node.func, ast.Name) and node.func.id in model_classes:
                        call_type = "model_class"
                        call_name = node.func.id
                    elif isinstance(node.func, ast.Attribute) and node.func.attr in model_classes:
                        call_type = "model_class"
                        call_name = node.func.attr

                    # Model method calls
                    elif isinstance(node.func, ast.Attribute) and node.func.attr in dl_methods:
                        call_type = "dl_method"
                        call_name = node.func.attr

                        # Special handling for fit method callbacks
                        if node.func.attr == 'fit':
                            handle_fit_callbacks(node)

                    # Optimizer initialization
                    elif isinstance(node.func, ast.Name) and node.func.id in optimizer_classes:
                        call_type = "optimizer"
                        call_name = node.func.id

                    # Direct callback initialization
                    elif (isinstance(node.func, ast.Name) and node.func.id in callback_classes) or \
                         (isinstance(node.func, ast.Attribute) and node.func.attr in callback_classes):
                        call_type = "callback"
                        call_name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr

                        # Process this callback directly
                        if handle_callback_node(node):
                            modified = True
                            continue

                    if not call_type:
                        continue

                    # Randomly select hyperparameters to modify/remove
                    if len(node.keywords) <= 8:
                        num_to_modify = max(1, int(len(node.keywords) * modification_ratio))
                    else:
                        num_to_modify = 4

                    params_to_modify = random.sample(node.keywords, num_to_modify)

                    params = ", ".join([keyword.arg for keyword in params_to_modify if isinstance(keyword.arg, str)])

                    if modification_type == 'remove':
                        logger.debug(f"Removing {call_type} '{call_name}' with parameters: {params}")
                    else:
                        logger.debug(f"Modifying {call_type} '{call_name}' with parameters: {params}")

                    # Process selected parameters
                    for keyword in params_to_modify:
                        if modification_type == 'remove':
                            # Skip essential parameters
                            if is_essential_param(keyword.arg, call_name):
                                continue
                            node.keywords.remove(keyword)
                            modified = True
                        else:  # modify
                            if modify_parameter(keyword, call_name):
                                modified = True

                    if modified:
                        mutation_data = {
                            'notebook_id': notebook_id,
                            'cell': idx,
                            'line': node.lineno,
                            'details': f'modification_ratio: {modification_ratio}, Modifying {call_type} "{call_name}" with parameters: {params}',
                            'code': str(CONFIG_MUTANTS[f"{modification_type}_hyperparameters"])
                        }
                        all_mutations.append(mutation_data)

                # Check for callback in variable assignments
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(node.value, ast.Call):
                            call_value = node.value
                            if (isinstance(call_value.func, ast.Name) and call_value.func.id in callback_classes) or \
                               (isinstance(call_value.func, ast.Attribute) and call_value.func.attr in callback_classes):
                                if find_and_modify_callbacks(call_value, node):
                                    modified = True

            cell.source = ast.unparse(tree) if modified else original_source

        if all_mutations:
            file_exists = os.path.isfile(os.path.join(res_dir, 'mutation_performed.csv'))
            mutant_df_log = pd.DataFrame(all_mutations)
            mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

            mutant_df_log.to_csv(
                os.path.join(res_dir, 'mutation_performed.csv'),
                mode='a',
                header=not file_exists,
                index=False
            )

    csv_path = os.path.join(res_dir, 'mutation_performed.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df_deduped = df.drop_duplicates()

        if len(df) != len(df_deduped):
            logger.debug(f"Dropped {len(df) - len(df_deduped)} duplicate mutation entries")
            df_deduped.to_csv(csv_path, index=False)
    return (notebook, modified)

def swap_metrics(notebook, logger, res_dir, mutant_num=1):
    logger.debug(f"######### METRIC_SWAP_LOGS #########")
    changed = False
    existing_imports = extract_existing_imports(notebook)
    missing_imports = set()

    mutant_log_df = pd.DataFrame(columns=['cell', 'line', 'details', 'code'])

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            try:
                tree = ast.parse(cell.source)
                replacer = MetricReplacer(logger, index, existing_imports, missing_imports)
                new_tree = replacer.visit(tree)
                if replacer.changed:
                    cell.source = ast.unparse(new_tree)
                    changed = True
                    for swap in replacer.swaps_performed:
                        swap_log = pd.DataFrame({
                            'cell': [index],
                            'line': [swap['line']],
                            'details': [f"Swapped {swap['old_value']} → {swap['new_value']}"],
                            'code': [str(CONFIG_MUTANTS["metric_swap"])]
                        })
                        mutant_log_df = pd.concat([mutant_log_df, swap_log], ignore_index=True)
            except SyntaxError:
                logger.warning(f"Skipping cell {index} due to syntax error.")

    # Add missing imports if necessary
    notebook = add_missing_imports(notebook, missing_imports)

    if not mutant_log_df.empty:
        mutant_log_df = mutant_log_df.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

        os.makedirs(res_dir, exist_ok=True)
        log_path = os.path.join(res_dir, 'mutation_performed.csv')
        mutant_log_df.to_csv(log_path, index=False)

    return [(notebook, changed)]

def label_error(notebook, logger, res_dir, mutant_num=1, error_ratio=0.15):
    logger.debug(f"######### LABEL_ERROR_LOGS #########")
    changed = False  # Track if any modification is made

    Y_var = None
    label_slice = None
    train_test_cell, train_test_line = None, None

    # First pass
    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            tree = ast.parse(cell.source)

            # Identify train_test_split() calls and track assigned variables
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name) and node.value.func.id == "train_test_split":
                        train_test_cell = idx
                        train_test_line = node.lineno

                        args = node.value.args
                        if len(args) >= 2:
                            if isinstance(args[1], ast.Name):
                                Y_var = args[1].id
                            elif isinstance(args[1], ast.Subscript):
                                if isinstance(args[1].slice, ast.List):
                                    label_slice = args[1].slice.elts[0].value
                                elif isinstance(args[1].slice, ast.Constant):
                                    label_slice = args[1].slice.value

                        logger.debug(f'{Y_var}, {label_slice}')
                        break

    # Second pass for identifying label slice
    if not label_slice:
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                tree = ast.parse(cell.source)

                # Identify Y assignment
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == Y_var:

                        if isinstance(node.value, ast.Subscript):
                            if isinstance(node.value.slice, ast.List):
                                label_slice = node.value.slice.elts[0].value
                            elif isinstance(node.value.slice, ast.Constant):
                                label_slice = node.value.slice.value

                        logger.debug(f'{Y_var}, {label_slice}')
                        break

    if label_slice:
        for idx, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                tree = ast.parse(cell.source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in PANDAS_READERS and isinstance(node.args[0], ast.Constant):
                        file_path = node.args[0].value

                        try:
                            df = eval(ast.unparse(node))
                        except Exception as e:
                            logger.debug(f"Could not open source dataset, due to following:")
                            logger.debug(f"{str(e)}")
                            break

                        try:
                            y = df[label_slice]
                        except:
                            continue

                        unique_labels = np.unique(y)
                        if isinstance(unique_labels[0], np.floating):
                            break
                        else:
                            logger.debug(f"Modifying {file_path} to introduce label errors with {len(unique_labels)} labels...")
                            num_errors = max(1, int(len(y) * error_ratio))
                            error_indices = np.random.choice(len(y), num_errors, replace=False)

                            for i in error_indices:
                                original_label = y.iloc[i]
                                new_label = original_label
                                while (new_label == original_label):
                                    new_label = np.random.choice(unique_labels)

                                df.loc[i, label_slice] = new_label

                        mod_path = get_inc_fname(node.args[0].value, '_with_label_errors')

                        new_fname = str(mod_path)

                        df.to_csv(mod_path, index=False)

                        mutant_df_log = pd.DataFrame()
                        mutant_df_log['cell'] = [idx]
                        mutant_df_log['line'] = [node.lineno]
                        mutant_df_log['details'] = [f'error_ratio: {error_ratio}, Modified indices: {str(error_indices)}']
                        mutant_df_log['code'] = [str(CONFIG_MUTANTS['label_errors'])]
                        mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

                        mutant_df_log.to_csv(os.path.join(res_dir, 'mutation_performed.csv'), index=False)

                        node.args[0].value = new_fname
                        changed = True
                        break

                cell.source = ast.unparse(tree)
    elif Y_var:
        cell = notebook.cells[train_test_cell]
        tree = ast.parse(cell.source)

        mutation_code = f"""
import numpy as np
unique_labels = np.unique({Y_var})

if not isinstance(unique_labels[0], np.floating):
    num_errors = max(1, int(len({Y_var}) * {error_ratio}))
    error_indices = np.random.choice(len({Y_var}), num_errors, replace=False)

for i in error_indices:
    original_label = {Y_var}.iloc[i]
    new_label = original_label
    while (new_label == original_label):
        new_label = np.random.choice(unique_labels)

    {Y_var}.iloc[i] = new_label
"""
        mutation_ast = ast.parse(mutation_code)

        # Insert the mutation code before train-test-split
        new_body = []
        for stmt in tree.body:
            new_body.append(stmt)

            # Check if the statement is an assignment using train_test_split
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                var_name = None
                if isinstance(stmt.value.func, ast.Name):
                    var_name = stmt.value.func.id
                elif isinstance(stmt.value.func, ast.Attribute):
                    var_name = stmt.value.func.attr

                if var_name == "train_test_split":
                    logger.debug(f"Inserting after: {ast.unparse(stmt)}")
                    new_body = new_body[:-1] + mutation_ast.body + [stmt]   # Ensure correct ordering

        tree.body = new_body  # Replace module body with modified body

        notebook.cells[train_test_cell].source = ast.unparse(tree)

        mutant_df_log = pd.DataFrame()
        mutant_df_log['cell'] = [train_test_cell]
        mutant_df_log['line'] = [train_test_line]
        mutant_df_log['details'] = [f'{Y_var}']
        mutant_df_log['code'] = [str(CONFIG_MUTANTS['label_errors'])]
        mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

        mutant_df_log.to_csv(os.path.join(res_dir, 'mutation_performed.csv'), index=False)

        changed = True

    else:
        df_srcs = detect_dataframe_sources(notebook, logger)

        for df_name in df_srcs.keys():
            if "label" in df_srcs[df_name]:
                src = df_srcs[df_name]
                creation_line = src['creation_line']

                cell_idx = src['cell_idx']
                cell = notebook.cells[cell_idx]

                tree = ast.parse(cell.source)

                mutation_code = f"""
import numpy as np
unique_labels = np.unique({src["label"]})

if not isinstance(unique_labels[0], np.floating):
    num_errors = max(1, int(len({src["label"]}) * {error_ratio}))
    error_indices = np.random.choice(len({src["label"]}), num_errors, replace=False)

    for i in error_indices:
        original_label = {src["label"]}[i]
        new_label = original_label
        while (new_label == original_label):
            new_label = np.random.choice(unique_labels)

        {src["label"]}[i] = new_label
"""
                mutation_ast = ast.parse(mutation_code)

                # Insert the mutation code after the DataFrame creation
                new_body = []
                for stmt in tree.body:
                    new_body.append(stmt)
                    if hasattr(stmt, 'lineno') and stmt.lineno == creation_line:
                        # Add mutation code
                        new_body.extend(mutation_ast.body)

                        # Log mutation details
                        mutant_df_log = pd.DataFrame()
                        mutant_df_log['cell'] = [cell_idx]
                        mutant_df_log['line'] = [creation_line]
                        mutant_df_log['details'] = [f"Label: {src['label']}"]
                        mutant_df_log['code'] = [CONFIG_MUTANTS['label_errors']]

                        mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

                        mutant_df_path = os.path.join(res_dir, 'mutation_performed.csv')
                        if os.path.exists(mutant_df_path):
                            prev_df = pd.read_csv(mutant_df_path)
                            new_df = pd.concat([prev_df, mutant_df_log], ignore_index=True)
                            new_df.to_csv(mutant_df_path, index=False)
                        else:
                            mutant_df_log.to_csv(mutant_df_path, index=False)

                        changed = True

                tree.body = new_body

                cell.source = ast.unparse(tree)
                changed = True
                break

    return [(notebook, changed)]

import copy
import nbformat

def data_shift(notebook, logger, res_dir, mutant_num=1):
    logger.debug(f"######### DATA_SHIFT_LOGS #########")
    all_mutants = []

    train_test_calls = []

    # First pass — find all train_test_split() calls
    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        try:
            tree = ast.parse(cell.source)
        except:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name) and node.value.func.id == "train_test_split":
                    args = node.value.args
                    if len(args) < 2:
                        continue

                    try:
                        X_var = args[0].id
                        Y_var = args[1].id
                    except AttributeError:
                        continue

                    if isinstance(node.targets[0], ast.Tuple) and len(node.targets[0].elts) == 4:
                        try:
                            targets = node.targets[0].elts
                            X_train, X_test, Y_train, Y_test = [elt.id for elt in targets]
                        except AttributeError:
                            continue

                        train_test_calls.append({
                            "cell_idx": idx,
                            "lineno": node.lineno,
                            "X_train": X_train,
                            "X_test": X_test,
                            "Y_train": Y_train,
                            "Y_test": Y_test
                        })
                        logger.debug(f"Found split: {X_train}, {X_test}, {Y_train}, {Y_test} in cell {idx}")

    for i, split in enumerate(train_test_calls):
        mutated_notebook = copy.deepcopy(notebook)
        cell = mutated_notebook.cells[split['cell_idx']]

        try:
            tree = ast.parse(cell.source)
        except:
            continue

        mutation_code = f"""
from scipy.stats import spearmanr
correlations = {{}}
for col in {split['X_train']}.columns:
    corr, _ = spearmanr({split['X_train']}[col], {split['Y_train']})
    correlations[col] = abs(corr)

most_correlated_feature = max(correlations, key=correlations.get)
{split['X_test']}[most_correlated_feature] = np.random.permutation({split['X_test']}[most_correlated_feature].values)
"""
        mutation_ast = ast.parse(mutation_code)

        # Inject mutation code after train_test_split
        new_body = []
        inserted = False
        for stmt in tree.body:
            new_body.append(stmt)
            if not inserted and isinstance(stmt, ast.Assign):
                if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name):
                    if stmt.value.func.id == "train_test_split":
                        logger.debug(f"Inserting after line {split['lineno']} in cell {split['cell_idx']}")
                        new_body.extend(mutation_ast.body)
                        inserted = True

        tree.body = new_body
        mutated_notebook.cells[split['cell_idx']].source = ast.unparse(tree)

        # Save mutated notebook
        mutated_path = os.path.join(res_dir, f"data_shift_{i}.ipynb")
        with open(mutated_path, 'w', encoding='utf-8') as f:
            nbformat.write(mutated_notebook, f)

        # Log mutation
        mutant_df_log = pd.DataFrame()
        mutant_df_log['cell'] = [split['cell_idx']]
        mutant_df_log['line'] = [split['lineno']]
        mutant_df_log['details'] = [f'X={split["X_train"]}, Y={split["Y_train"]}']
        mutant_df_log['code'] = [str(CONFIG_MUTANTS['data_shift'])]
        mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

        log_path = os.path.join(res_dir, 'mutation_performed.csv')
        if os.path.exists(log_path):
            prev = pd.read_csv(log_path)
            pd.concat([prev, mutant_df_log], ignore_index=True).to_csv(log_path, index=False)
        else:
            mutant_df_log.to_csv(log_path, index=False)

        all_mutants.append((mutated_notebook, True))

    return all_mutants


def preprocessing_leakage(notebook, logger, res_dir, mutant_num=1):
    logger.debug(f"######### PREPROC_LEAKAGE_LOGS #########")
    changed = False  # Track if any modification is made

    class API_Remover(ast.NodeTransformer):
        def __init__(self, logger, var_name):
            super().__init__()
            self.logger = logger
            self.var_name = var_name

        def visit_Assign(self, node):
            """
            Removes assignments where `var_name.transform()` or `var_name.fit_transform()` is called,
            but **keeps** assignments where `var_name` is initialized (e.g., `sc = StandardScaler()`).
            """
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                if node.value.func.attr in ['fit', 'transform', 'fit_transform'] and node.value.func.value.id == self.var_name:
                    self.logger.debug(f"Removed {ast.unparse(node)}")
                    return None  # Remove this line

            return node

    X_var, Y_var = None, None
    train_test_cell, train_test_line = -1, -1

    SCALING_APIS = ['MaxAbsScaler', 'MinMaxScaler', 'RobustScaler', 'StandardScaler']
    FEATURE_SELECTION_APIS = ['VarianceThreshold', 'SelectKBest', 'SelectPercentile', 'GenericUnivariateSelect', 'RFE', 'RFECV']

    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            tree = ast.parse(cell.source)

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Name) and node.func.id == 'train_test_split') or (isinstance(node.func, ast.Attribute) and node.func.attr == 'train_test_split'):
                        args = node.args
                        if len(args) >= 2:
                            if isinstance(args[0], ast.Name) and isinstance(args[1], ast.Name):
                                X_var = args[0].id
                                Y_var = args[1].id
                                train_test_cell = idx
                                train_test_line = node.lineno
                                logger.debug(f"X: {X_var}, y: {Y_var}, cell: {train_test_cell}, line: {train_test_line}, node: {ast.unparse(node)}")

    if train_test_cell == -1:
        return [(notebook, False)]

    scalers_features_sel_vars = {}

    lines = []

    # Second pass
    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            tree = ast.parse(cell.source)

            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                    func_name = None
                    if isinstance(node.value.func, ast.Name):
                        func_name = node.value.func.id
                    elif isinstance(node.value.func, ast.Attribute):
                        func_name = node.value.func.attr

                    if func_name in SCALING_APIS or func_name in FEATURE_SELECTION_APIS:
                        var = node.targets[0].id
                        scalers_features_sel_vars[var] = ast.unparse(node)
                        logger.debug(f"{ast.unparse(node)}")

    if not scalers_features_sel_vars:
        return [(notebook, False)]

    api_cell, api_line = -1, -1
    api_name = None
    var_name = None
    preproc_func_name = None

    # Third pass to identify API usage location
    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            tree = ast.parse(cell.source)

            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['fit_transform', 'transform'] and isinstance(node.func.value, ast.Name) and node.func.value.id in scalers_features_sel_vars.keys():
                        api_cell = idx
                        api_line = node.lineno
                        var_name = node.func.value.id
                        def_line = scalers_features_sel_vars[var_name]
                        api_name = node.func.attr

                        tmp_tree = ast.parse(def_line)

                        for n in ast.walk(tmp_tree):
                            if isinstance(n, ast.Call):
                                if isinstance(n.func, ast.Name):
                                    preproc_func_name = n.func.id
                                elif isinstance(n.func, ast.Attribute):
                                    preproc_func_name = n.func.attr
                                break

                        if preproc_func_name in SCALING_APIS:
                            lines.append(f"from sklearn.preprocessing import {preproc_func_name}")
                        else:
                            lines.append(f"from sklearn.feature_selection import {preproc_func_name}")
                        lines.append(def_line)
                        logger.debug(f"{var_name}")

                        new_tree = API_Remover(logger, var_name).visit(tree)
                        ast.fix_missing_locations(new_tree)
                        cell.source = ast.unparse(new_tree)
                        break

    if api_cell == -1:
        return [(notebook, False)]

    logger.debug(f"{train_test_cell}, {api_cell}, {train_test_line}, {api_line}")

    # Introduce preprocessing leakage
    if api_cell > train_test_cell or (api_cell == train_test_cell and api_line > train_test_line):
        mutant_cell = notebook.cells[train_test_cell].source
        if api_name == 'transform':
            lines.append(f"{X_var} = {var_name}.fit({X_var})")

        lines.append(f"{X_var} = {var_name}.{api_name}({X_var})")
        logger.debug(f"New lines: {str(lines)}")

        lines = list(OrderedDict.fromkeys(lines))

        mutation_code = "\n".join(lines)
        mutation_ast = ast.parse(mutation_code)

        # Insert the mutation code before train_test_split
        tree = ast.parse(mutant_cell)  # Parse code into AST
        new_body = []  # Stores modified code

        for stmt in tree.body:
            new_body.append(stmt)
            # Check if the statement is an assignment using train_test_split
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                var_name = None
                if isinstance(stmt.value.func, ast.Name):
                    var_name = stmt.value.func.id
                elif isinstance(stmt.value.func, ast.Attribute):
                    var_name = stmt.value.func.attr

                if var_name == "train_test_split":
                    logger.debug(f"Inserting before: {ast.unparse(stmt)}")
                    new_body = new_body[:-1] + mutation_ast.body + [stmt]  # Ensure correct ordering

        tree.body = new_body  # Replace module body with modified body

        notebook.cells[train_test_cell].source = ast.unparse(tree)

        logger.debug(f"New cell: {notebook.cells[train_test_cell].source}")
        changed = True
        mutant_df_log = pd.DataFrame()
        mutant_df_log['cell'] = [train_test_cell]
        mutant_df_log['line'] = [train_test_line]
        mutant_df_log['details'] = [f'Leakage induced via: {preproc_func_name}']
        mutant_df_log['code'] = [str(CONFIG_MUTANTS['preproc_data_leakage'])]
        mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

        mutant_df_log.to_csv(os.path.join(res_dir, 'mutation_performed.csv'), index=False)

    return [(notebook, changed)]

def find_model_definitions(notebook):
    model_defs = []

    for cell_idx, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        try:
            tree = ast.parse(cell.source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Matches: model = Sequential([...])
                if isinstance(node.func, ast.Name) and node.func.id == "Sequential":
                    model_defs.append(("sequential_call", cell_idx))

                # Matches: model = tf.keras.Sequential([...])
                elif isinstance(node.func, ast.Attribute) and node.func.attr == "Sequential":
                    model_defs.append(("sequential_call", cell_idx))

                elif isinstance(node.func, ast.Attribute) and node.func.attr == "add":
                    model_defs.append(("add_call", cell_idx))

            elif isinstance(node, ast.ClassDef):
                if any((isinstance(base, ast.Attribute) and base.attr == "Module") or
                       (isinstance(base, ast.Name) and base.id == "Module")
                       for base in node.bases):
                    model_defs.append(("class_model", cell_idx))

    return model_defs

class TensorflowLayerInserter(ast.NodeTransformer):
    def __init__(self):
        self.inserted = False
        self.layers = [
            "ReLU()",
            "BatchNormalization()",
            "Dropout(0.25)"
        ]
        self.selected_layer = None  # Store selected layer to repeat
        self.insert_pos = -1

    def get_layer_code(self):
        if not self.selected_layer:
            self.selected_layer = random.choice(self.layers)
        return f"keras.layers.{self.selected_layer}"

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            if node.value.func.attr == "add" and not self.inserted:
                model_var = ast.unparse(node.value.func.value)
                layer_code = self.get_layer_code()
                first_add = ast.parse(f"{model_var}.add({layer_code})").body[0]
                second_add = ast.parse(f"{model_var}.add({layer_code})").body[0]
                self.inserted = True
                return [node, first_add, second_add]
        return node

    def visit_Call(self, node):
        if self.inserted:
            return (node)

        if ((isinstance(node.func, ast.Name) and node.func.id == "Sequential") or
            (isinstance(node.func, ast.Attribute) and node.func.attr == "Sequential")):
            if node.args and isinstance(node.args[0], ast.List):
                insert_pos = random.randint(1, len(node.args[0].elts) - 1)
                layer_ast1 = ast.parse(self.get_layer_code()).body[0].value
                layer_ast2 = ast.parse(self.get_layer_code()).body[0].value
                node.args[0].elts.insert(insert_pos, layer_ast2)
                node.args[0].elts.insert(insert_pos, layer_ast1)
                self.inserted = True
                self.insert_pos = insert_pos

        return self.generic_visit(node)

class PyTorchClassMutator(ast.NodeTransformer):
    def __init__(self, layer_name="injected_layer"):
        self.layers = [
            "nn.ReLU()",
            "nn.Identity()",
            "nn.Dropout()",
            "nn.Tanh()"
        ]
        self.layer_name = layer_name
        self.layer_code = random.choice(self.layers)
        self.init_inserted = False
        self.forward_inserted = False
        self.insert_index = None

    def visit_ClassDef(self, node):
        is_module = any(
            (isinstance(base, ast.Attribute) and base.attr == "Module") or
            (isinstance(base, ast.Name) and base.id == "Module")
            for base in node.bases
        )

        if not is_module:
            return node

        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                if stmt.name == "__init__":
                    self._mutate_init(stmt)
                elif stmt.name == "forward":
                    self._mutate_forward(stmt)

        return node

    def _mutate_init(self, init_fn):
        inject_code1 = f"self.{self.layer_name}_1 = {self.layer_code}"
        inject_code2 = f"self.{self.layer_name}_2 = {self.layer_code}"
        inject_node1 = ast.parse(inject_code1).body[0]
        inject_node2 = ast.parse(inject_code2).body[0]

        for i in range(len(init_fn.body)):
            if isinstance(init_fn.body[i], ast.Return):
                init_fn.body.insert(i, inject_node2)
                init_fn.body.insert(i, inject_node1)
                self.init_inserted = True
                return

        init_fn.body.extend([inject_node1, inject_node2])
        self.init_inserted = True

    def _mutate_forward(self, forward_fn):
        insert_index = -1
        target_name = ''
        stmt_count = 0

        for i, stmt in enumerate(forward_fn.body):
            if isinstance(stmt, ast.Assign):
                targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                stmt_count += 1
                if targets:
                    target_name = targets[0]

        self.insert_index = max(0, np.floor(stmt_count / 2))

        if target_name:
            inject_code1 = f"{target_name} = self.{self.layer_name}_1({target_name})"
            inject_code2 = f"{target_name} = self.{self.layer_name}_2({target_name})"
            inject_node1 = ast.parse(inject_code1).body[0]
            inject_node2 = ast.parse(inject_code2).body[0]
            forward_fn.body.insert(insert_index + 1, inject_node2)
            forward_fn.body.insert(insert_index + 1, inject_node1)

        self.forward_inserted = True

def deep_layer_insertion(notebook, logger, res_dir, mutant_num=1):
    logger.debug(f"######### DEEP_LAYER_INSERTION_LOGS #########")
    model_defs = find_model_definitions(notebook)
    changed = False
    mutated_count = 0

    logger.debug(model_defs)
    if len(model_defs) == 0:
        logger.debug("No model definitions found.")
        return [(notebook, changed)]

    for model_type, cell_idx in model_defs:
        if mutated_count >= 4:
            break

        insert_pos, new_layers = None, None
        cell = notebook.cells[cell_idx]
        try:
            tree = ast.parse(cell.source)

            if model_type == "sequential_call" or model_type == "add_call":
                transformer = TensorflowLayerInserter()
                new_tree = transformer.visit(tree)

                if transformer.inserted:
                    org_src = cell.source
                    cell.source = ast.unparse(new_tree)
                    logger.debug(f"Inserted layer into cell {cell_idx}")
                    logger.debug("Original")
                    logger.debug(org_src)
                    logger.debug("Mutated")
                    logger.debug(cell.source)
                    insert_pos = transformer.insert_pos
                    new_layers = transformer.selected_layer
                    changed = True
                    mutated_count += 1
                    yield notebook, changed
                    cell.source = org_src

            else:
                transformer = PyTorchClassMutator()
                new_tree = transformer.visit(tree)

                if transformer.forward_inserted:
                    org_src = cell.source
                    cell.source = ast.unparse(new_tree)
                    logger.debug(f"Inserted layer into cell {cell_idx}")
                    logger.debug("Original")
                    logger.debug(org_src)
                    logger.debug("Mutated")
                    logger.debug(cell.source)
                    insert_pos = transformer.insert_index
                    new_layers = transformer.layer_code
                    changed = True
                    mutated_count += 1
                    yield (notebook, changed)
                    cell.source = org_src

            if changed:
                changed = False
                mutant_df_log = pd.DataFrame()
                mutant_df_log['cell'] = [cell_idx]
                mutant_df_log['line'] = [insert_pos]
                mutant_df_log['details'] = [f"{new_layers} * 2"]
                mutant_df_log['code'] = [f"{CONFIG_MUTANTS['deep_layer_insertion']}"]

                mutant_df_log = mutant_df_log.map(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

                mutant_df_path = os.path.join(res_dir, 'mutation_performed.csv')
                if os.path.exists(mutant_df_path):
                    prev_df = pd.read_csv(mutant_df_path)
                    new_df = pd.concat([prev_df, mutant_df_log], ignore_index=True)
                    new_df.to_csv(mutant_df_path, index=False)
                else:
                    mutant_df_log.to_csv(mutant_df_path, index=False)

        except Exception as e:
            continue

    return [(notebook, changed)]

class TensorflowLayerRemover(ast.NodeTransformer):
    def __init__(self):
        self.removed = False
        self.removed_layer = None
        self.remove_index = None
        self.removable = ["ReLU", "Dropout", "BatchNormalization"]

    def visit_Expr(self, node):
        if self.removed:
            return node

        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            if node.value.func.attr == "add":
                try:
                    inner_call = node.value.args[0]
                    if isinstance(inner_call, ast.Call) and hasattr(inner_call.func, "id"):
                        if inner_call.func.id in self.removable:
                            self.removed_layer = inner_call.func.id
                            self.removed = True
                            return None
                except:
                    pass
        return node

    def visit_Call(self, node):
        if self.removed:
            return (node)

        if (isinstance(node.func, ast.Name) and node.func.id == "Sequential" or
            (isinstance(node.func, ast.Attribute) and node.func.attr == "Sequential")):
            if node.args and isinstance(node.args[0], ast.List):
                for i, elt in enumerate(node.args[0].elts[1:-1], 1):
                    if isinstance(elt, ast.Call):
                        func_name = None
                        if isinstance(elt.func, ast.Name):
                            func_name = elt.func.id
                        elif isinstance(elt.func, ast.Attribute):
                            func_name = elt.func.attr

                        if func_name in self.removable:
                            self.removed_layer = func_name
                            self.remove_index = i
                            del node.args[0].elts[i]
                            self.removed = True
                            break

        return self.generic_visit(node)

class PyTorchLayerRemover(ast.NodeTransformer):
    def __init__(self):
        self.removed = False
        self.layer_name = None
        self.forward_line = None
        self.init_line = None
        self.removable = ["nn.ReLU", "nn.Identity", "nn.Dropout", "nn.Tanh"]

    def visit_ClassDef(self, node):
        is_module = any(
            (isinstance(base, ast.Attribute) and base.attr == "Module") or
            (isinstance(base, ast.Name) and base.id == "Module")
            for base in node.bases
        )

        if not is_module:
            return node

        init_func = None
        forward_func = None

        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                init_func = stmt
            elif isinstance(stmt, ast.FunctionDef) and stmt.name == "forward":
                forward_func = stmt

        if not init_func or not forward_func:
            return node

        # Remove layer from __init__
        for i, stmt in enumerate(init_func.body):
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                try:
                    value = ast.unparse(stmt.value)
                    if any(r in value for r in self.removable):
                        self.layer_name = ast.unparse(stmt.targets[0])
                        self.init_line = i
                        del init_func.body[i]
                        self.removed = True
                        break
                except:
                    continue

        # Remove corresponding call from forward
        if self.removed and self.layer_name:
            for i, stmt in enumerate(forward_func.body):
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                    if ast.unparse(stmt.value.func).strip() == f"self.{self.layer_name}":
                        del forward_func.body[i]
                        self.forward_line = i
                        break

        return node

def deep_layer_removal(notebook, logger, res_dir, mutant_num=1):
    logger.debug(f"######### DEEP_LAYER_REMOVAL_LOGS #########")
    model_defs = find_model_definitions(notebook)
    changed = False
    mutated_count = 0

    logger.debug(model_defs)

    if len(model_defs) == 0:
        logger.debug("No model definitions found.")
        return [(notebook, changed)]

    for model_type, cell_idx in model_defs:
        if mutated_count >= 4:
            break

        cell = notebook.cells[cell_idx]
        tree = ast.parse(cell.source)

        if model_type in ("sequential_call", "add_call"):
            transformer = TensorflowLayerRemover()
            new_tree = transformer.visit(tree)

            if transformer.removed:
                org_src = cell.source
                cell.source = ast.unparse(new_tree)
                logger.debug(f"Removed layer from cell {cell_idx}")
                changed = True
                mutated_count += 1

                mutant_df_log = pd.DataFrame([{
                    "cell": cell_idx,
                    "line": transformer.remove_index,
                    "details": f"Removed keras.layers.{transformer.removed_layer}",
                    "code": CONFIG_MUTANTS['deep_layer_removal']
                }])
                mutant_df_log.to_csv(os.path.join(res_dir, 'mutation_performed.csv'), mode='a', header=not os.path.exists(os.path.join(res_dir, 'mutation_performed.csv')))

                yield notebook, changed
                cell.source = org_src

        elif model_type == "class_model":
            transformer = PyTorchLayerRemover()
            new_tree = transformer.visit(tree)

            if transformer.removed:
                org_src = cell.source
                cell.source = ast.unparse(new_tree)
                logger.debug(f"Removed layer from cell {cell_idx}")
                changed = True
                mutated_count += 1

                mutant_df_log = pd.DataFrame([{
                    "cell": cell_idx,
                    "line": transformer.forward_line,
                    "details": f"Removed {transformer.layer_name}",
                    "code": CONFIG_MUTANTS['deep_layer_removal']
                }])
                mutant_df_log.to_csv(os.path.join(res_dir, 'mutation_performed.csv'), mode='a', header=not os.path.exists(os.path.join(res_dir, 'mutation_performed.csv')))

                yield (notebook, changed)
                cell.source = org_src

    return [(notebook, changed)]

class TensorflowLayerChanger(ast.NodeTransformer):
    def __init__(self):
        self.changed = False
        self.replaced_layer = None
        self.new_layer = None
        self.change_index = None
        self.replacements = {
            "ReLU": "Tanh",
            "Dropout": "Dropout(0.3)",
            "BatchNormalization": "LayerNormalization"
        }

    def visit_Expr(self, node):
        if self.changed:
            return node

        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            if node.value.func.attr == "add":
                try:
                    inner_call = node.value.args[0]
                    if isinstance(inner_call, ast.Call):
                        func_name = None
                        if isinstance(inner_call.func, ast.Name):
                            func_name = inner_call.func.id
                        elif isinstance(inner_call.func, ast.Attribute):
                            func_name = inner_call.func.attr

                        if func_name in self.replacements:
                            self.replaced_layer = func_name
                            self.new_layer = self.replacements[func_name]
                            new_call = ast.parse(f"{node.value.func.value.id}.add(keras.layers.{self.new_layer})").body[0]
                            self.changed = True
                            return new_call
                except:
                    pass
        return node

    def visit_Call(self, node):
        if self.changed:
            return (node)

        if ((isinstance(node.func, ast.Name) and node.func.id == "Sequential") or
            (isinstance(node.func, ast.Attribute) and node.func.attr == "Sequential")):
            if node.args and isinstance(node.args[0], ast.List):
                for i, elt in enumerate(node.args[0].elts[1:-1], 1):
                    if isinstance(elt, ast.Call):
                        old_layer = None
                        if isinstance(elt.func, ast.Name):
                            old_layer = elt.func.id
                        elif isinstance(elt.func, ast.Attribute):
                            old_layer = elt.func.attr

                        if old_layer in self.replacements:
                            self.replaced_layer = old_layer
                            new_layer = self.replacements[old_layer]
                            self.new_layer = new_layer
                            self.change_index = i
                            new_node = ast.parse(f"keras.layers.{new_layer}").body[0].value
                            node.args[0].elts[i] = new_node
                            self.changed = True
                            break
        return self.generic_visit(node)

class PyTorchLayerChanger(ast.NodeTransformer):
    def __init__(self):
        self.replaced = False
        self.replaced_layer = None
        self.new_layer = None
        self.forward_index = None
        self.init_index = None
        self.replacements = {
            "nn.ReLU": "nn.Tanh",
            "nn.Tanh": "nn.Sigmoid",
            "nn.Identity": "nn.ReLU",
            "nn.Dropout": "nn.Dropout(p=0.3)"
        }

    def visit_ClassDef(self, node):
        is_module = any(
            (isinstance(base, ast.Attribute) and base.attr == "Module") or
            (isinstance(base, ast.Name) and base.id == "Module")
            for base in node.bases
        )
        if not is_module:
            return node

        init_fn = None
        forw_fn = None
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                if stmt.name == "__init__":
                    init_fn = stmt
                elif stmt.name == "forward":
                    forw_fn = stmt

        if init_fn:
            for i, stmt in enumerate(init_fn.body):
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                    try:
                        old_code = ast.unparse(stmt.value)
                        for old, new in self.replacements.items():
                            if old in old_code:
                                new_code = old_code.replace(old, new)
                                stmt.value = ast.parse(new_code).body[0].value
                                self.replaced = True
                                self.replaced_layer = old
                                self.new_layer = new
                                self.init_index = i
                                break
                        if self.replaced:
                            break
                    except:
                        continue
        return node

def deep_layer_change(notebook, logger, res_dir, mutant_num=1):
    logger.debug(f"######### DEEP_LAYER_CHANGE_LOGS #########")
    model_defs = find_model_definitions(notebook)
    changed = False
    mutated_count = 0

    logger.debug(model_defs)

    for model_type, cell_idx in model_defs:
        if mutated_count >= 4:
            break

        cell = notebook.cells[cell_idx]
        try:
            tree = ast.parse(cell.source)

            if model_type in ("sequential_call", "add_call"):
                transformer = TensorflowLayerChanger()
                new_tree = transformer.visit(tree)

                if transformer.changed:
                    org_src = cell.source
                    cell.source = ast.unparse(new_tree)
                    changed = True
                    mutated_count += 1

                    logger.debug(f"Changed layer type in cell {cell_idx}: {transformer.replaced_layer} → {transformer.new_layer}")
                    df = pd.DataFrame([{
                        "cell": cell_idx,
                        "line": transformer.change_index,
                        "details": f"Changed keras.layers.{transformer.replaced_layer} → keras.layers.{transformer.new_layer}",
                        "code": CONFIG_MUTANTS["deep_layer_change"]
                    }])

                    df.to_csv(os.path.join(res_dir, 'mutation_performed.csv'), mode='a', header=not os.path.exists(os.path.join(res_dir, 'mutation_performed.csv')))
                    yield (notebook, changed)
                    cell.source = org_src

            elif model_type == "class_model":
                transformer = PyTorchLayerChanger()
                new_tree = transformer.visit(tree)

                if transformer.replaced:
                    org_src = cell.source
                    cell.source = ast.unparse(new_tree)
                    changed = True
                    mutated_count += 1

                    logger.debug(f"Changed PyTorch layer type in cell {cell_idx}: {transformer.replaced_layer} → {transformer.new_layer}")
                    df = pd.DataFrame([{
                        "cell": cell_idx,
                        "line": transformer.init_index,
                        "details": f"Changed {transformer.replaced_layer} → {transformer.new_layer}",
                        "code": CONFIG_MUTANTS["deep_layer_change"]
                    }])
                    df.to_csv(os.path.join(res_dir, 'mutation_performed.csv'), mode='a', header=not os.path.exists(os.path.join(res_dir, 'mutation_performed.csv')))
                    yield (notebook, changed)
                    cell.source = org_src

        except Exception as e:
            logger.debug(f"Error in cell {cell_idx}: {e}")
            continue

    return [(notebook, changed)]

def generate_each_mutant(mutant_type, output_dir, notebook_fname, notebook_path, logger):
    logger.info(f"Generating mutants for {mutant_type}")

    each_mutant_count = 0
    notebook = load_notebook(notebook_path)

    mutant_output_dir = os.path.join(output_dir, f"{mutant_type}")
    os.makedirs(mutant_output_dir, exist_ok=True)

    mutant_log_dir = os.path.join(mutant_output_dir, "logs")
    os.makedirs(mutant_log_dir, exist_ok=True)
    res_dir = os.path.join(mutant_log_dir, f"{notebook_fname}_mutant_{mutant_type}_logs")
    os.makedirs(res_dir, exist_ok=True)

    # Check if the mutant count is less than MAX_MUTANT_COUNT
    mutant_notebooks_dir = os.path.join(mutant_output_dir, "mutant_notebooks")
    os.makedirs(mutant_notebooks_dir, exist_ok=True)

    unfinished_ite_list = get_unfinished_mutant_idx(mutant_notebooks_dir, MAX_MUTANT_COUNT, mutant_type, notebook_fname)
    if unfinished_ite_list:
        logger.info(f"Unfinished {mutant_type}: {unfinished_ite_list}")
        if mutant_type == "remove_hyperparameters" or mutant_type == "modify_hyperparameters":
            mutation_type = mutant_type.split("_")[0]
            unfinished_ite = len(unfinished_ite_list)
            log_file_path = os.path.join(res_dir, 'mutation_performed.csv')
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
            for i in range(unfinished_ite):
                mutant_idx = unfinished_ite_list.pop(0)
                # Generate a unique ID for this mutant
                notebook_id = f"{notebook_fname}_mutant_{mutation_type}_hyperparameters_{mutant_idx}"

                res_notebook, changed = modify_hyperparameters(
                    nbformat.from_dict(notebook),
                    logger,
                    res_dir,
                    modification_type=mutation_type,
                    notebook_id=notebook_id
                )

                if changed:
                    logger.debug(f"Mutation type: {mutation_type}, Mutation number: {mutant_idx}")
                    mutant_path = os.path.join(mutant_notebooks_dir, f"{notebook_fname}_mutant_{mutant_type}_{mutant_idx}.ipynb")
                    save_notebook(res_notebook, mutant_path)
                    logger.info(f"Generated mutant {mutant_idx} for {mutant_type} at {mutant_path}")


                    mutant_basename = os.path.basename(mutant_path)

                    update_df = pd.read_csv(log_file_path)
                    update_df.loc[update_df['notebook_id'] == notebook_id, 'notebook_id'] = mutant_basename
                    update_df.to_csv(log_file_path, index=False)

                    each_mutant_count += 1
        else:
            mutant_list = MUTANT_FUNC_MAP[mutant_type](nbformat.from_dict(notebook), logger, res_dir, len(unfinished_ite_list))

            try:

                for (res_notebook, changed) in mutant_list:
                    if changed:
                        if unfinished_ite_list:
                            mutant_idx = unfinished_ite_list.pop(0)
                            mutant_path = os.path.join(mutant_notebooks_dir, f"{notebook_fname}_mutant_{mutant_type}_{mutant_idx}.ipynb")
                            save_notebook(res_notebook, mutant_path)
                            logger.info(f"Generated mutant {mutant_idx} for {mutant_type} at {mutant_path}")
                            each_mutant_count += 1 # TODO: What is this mutant count?
                        else:
                            break
            except Exception as e:
                logger.error(f"{mutant_type}")
                logger.error(traceback.format_exc())

    else:
        logger.info(f"Found {MAX_MUTANT_COUNT} finished mutants. No need to generate more mutants.")
    return each_mutant_count

MUTANT_FUNC_MAP = {
        "outliers": introduce_outliers,
        "repetition": remove_and_repeat,
        "added_null": add_null,
        "remove_torch_eval": remove_eval,
        "remove_torch_zero_grad": remove_zero_grad,
        "remove_hyperparameters": modify_hyperparameters,
        "modify_hyperparameters": modify_hyperparameters,
        "label_errors": label_error,
        "preproc_data_leakage": preprocessing_leakage,
        "data_shift": data_shift,
        "metric_swap": swap_metrics,
        "deep_layer_insertion": deep_layer_insertion,
        "deep_layer_removal": deep_layer_removal,
        "deep_layer_change": deep_layer_change
    }

def generate_mutants(notebook_path, output_dir, notebook_fname):
    os.makedirs(output_dir, exist_ok=True)
    # Logging setup
    logs_dir = os.path.join(output_dir, f"A_logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger = setup_logger(os.path.join(logs_dir, 'gen_mutants.log'), 'gen_mutants.log')

    mutant_count = 0

    for mutant_type in MUTANT_FUNC_MAP:
        mutant_count += generate_each_mutant(mutant_type, output_dir, notebook_fname, notebook_path, logger)

if __name__ == "__main__":
    # input_notebook = sys.argv[1]
    # output_directory = "mutants" if not sys.argv[2] else sys.argv[2]

    # notebook_fname = input_notebook.split('/')[-1].split('.ipynb')[0]

    # nb = load_notebook(input_notebook)
    # generate_mutants(nb, output_directory, notebook_fname)

    parser = argparse.ArgumentParser()
    parser.add_argument('notebook', type=str, help='The path to the Jupyter notebook file')
    parser.add_argument('-o', required=True, help='Absolute path for mutants directory')

    args = parser.parse_args()
    notebook_w_assertions = args.notebook
    output_dir = args.o

    notebook_fname = get_notebook_name(notebook_w_assertions)
    mutation_output_dir = os.path.join(output_dir, f"mutation_{notebook_fname}")

    generate_mutants(notebook_w_assertions, mutation_output_dir, notebook_fname)
