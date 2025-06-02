import argparse
import os
import subprocess
import pandas as pd
import nbformat
import ast
from nbformat.v4 import new_code_cell
import csv
from datetime import datetime
import re
import json
import urllib.parse

import sys
import traceback
proj_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(proj_folder, '../'))

from utils.utils import setup_logger
from utils.pytest_run import run_missing_tests

import nbformat
from nbformat.validator import validate, ValidationError
from pathlib import Path

def is_valid_notebook(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            validate(nb)  
        return True
    except (nbformat.reader.NotJSONError, ValidationError, OSError) as e:
        return False



def get_ast_structure(code):
    try:
        tree = ast.parse(code)
        return tree
    except SyntaxError:
        return None

class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []
        self.variables = []
        self.assignments = []
    
    def visit_Call(self, node):
        # Capture function calls
        func_name = ast.dump(node.func)  # Function name
        num_args = len(node.args) 
        kwarg_names = {kw.arg for kw in node.keywords}
        self.functions.append((func_name, num_args, kwarg_names))
        self.generic_visit(node)

    def visit_Name(self, node):
        # Capture variable names
        self.variables.append(node.id)
        self.generic_visit(node)
        
    def visit_Assign(self, node):
        # Capture assignment targets
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assignments.append(target.id)
        self.generic_visit(node)

def extract_ast_elements(tree):
    visitor = CodeVisitor()
    visitor.visit(tree)
    return visitor.functions, visitor.variables, visitor.assignments

def remove_comments(code):
    """Remove comments from code while preserving line structure"""
    lines = []
    for line in code.split('\n'):
        # Only add non-empty lines after stripping whitespace
        if line.strip() and '#' not in line:
            lines.append(line)
    return '\n'.join(lines)

def is_code_match(old_code, new_code):
    """Compare code structure excluding comments and assertion statements"""
    # Remove comments from both code blocks
    old_code_no_comments = remove_comments(old_code)
    new_code_no_comments = remove_comments(new_code)
    
    # Filter out nbtest assertions from both code blocks for comparison
    old_code_filtered = '\n'.join([line for line in old_code_no_comments.split('\n') 
                                   if not line.strip().startswith('nbtest.assert')])
    new_code_filtered = '\n'.join([line for line in new_code_no_comments.split('\n') 
                                   if not line.strip().startswith('nbtest.assert')])
    
    old_tree = get_ast_structure(old_code_filtered)
    new_tree = get_ast_structure(new_code_filtered)
    
    if old_tree is None or new_tree is None:
        return False
    
    old_funcs, old_vars, old_assigns = extract_ast_elements(old_tree)
    new_funcs, new_vars, new_assigns = extract_ast_elements(new_tree)

    
    # Consider assigned variables as key indicators for matching
    if set(old_assigns) == set(new_assigns):
        # Compare key function calls
        if len(old_funcs) == len(new_funcs):
            for old_func, new_func in zip(sorted(old_funcs), sorted(new_funcs)):
                old_name, old_num_args, old_kwargs = old_func
                new_name, new_num_args, new_kwargs = new_func
                if old_name != new_name or old_num_args != new_num_args:
                    return False
            return True
    
    return False

def extract_main_code(code):
    """Extract the main code excluding assertions"""
    return '\n'.join([line for line in code.split('\n') 
                     if not line.strip().startswith('nbtest.assert')])

def get_cell_source(cell):
    """Extract source from a cell, handling different cell formats"""
    source = ''
    if hasattr(cell, 'source'):
        source = cell.source
    elif isinstance(cell, dict) and 'source' in cell:
        source = cell['source']
    
    # Handle list sources
    if isinstance(source, list):
        source = ''.join(source)
    
    return source

def get_cell_type(cell):
    """Get the cell type, handling different cell formats"""
    if hasattr(cell, 'cell_type'):
        return cell.cell_type
    elif isinstance(cell, dict) and 'cell_type' in cell:
        return cell['cell_type']
    return None

def extract_code_with_assertions(code):
    """Extract code lines with their associated assertions, maintaining order"""
    lines = code.split('\n')
    code_blocks = []
    current_block = []
    
    for line in lines:
        if line.strip().startswith('nbtest.assert'):
            # If we have accumulated code, save it with this assertion
            if current_block:
                code_blocks.append({
                    'code': '\n'.join(current_block),
                    'assertion': line
                })
                current_block = []
            else:
                # Assertion without preceding code (edge case)
                code_blocks.append({
                    'code': '',
                    'assertion': line
                })
        else:
            # Regular code line
            if line.strip():
                current_block.append(line)
    
    # Handle any remaining code
    if current_block:
        code_blocks.append({
            'code': '\n'.join(current_block),
            'assertion': None
        })
    
    return code_blocks

def reconstruct_code_with_assertions(old_code, latest_code_blocks):
    """Reconstruct code by matching old code patterns with assertion blocks"""
    old_lines = [line for line in old_code.split('\n') if line.strip() and '#' not in line]
    reconstructed_lines = []
    old_line_idx = 0
    
    for block in latest_code_blocks:
        block_code_lines = [line for line in block['code'].split('\n') if line.strip()]
        
        matched_old_lines = []
        temp_old_idx = old_line_idx
        
        for block_line in block_code_lines:
            # Look for similar line in old code (fuzzy matching)
            while temp_old_idx < len(old_lines):
                old_line = old_lines[temp_old_idx]
                
                # Simple similarity check
                if (extract_key_identifiers(block_line) == extract_key_identifiers(old_line) or
                    are_lines_similar(block_line, old_line)):
                    matched_old_lines.append(old_line)
                    temp_old_idx += 1
                    break
                else:
                    # If we can't match, include the old line anyway to preserve structure
                    matched_old_lines.append(old_line)
                    temp_old_idx += 1
                    break
            else:
                matched_old_lines.append(block_line)
        
        if matched_old_lines:
            reconstructed_lines.extend(matched_old_lines)
        
        if block['assertion']:
            reconstructed_lines.append(block['assertion'])
        
        old_line_idx = temp_old_idx
    
    # Add any remaining old lines
    while old_line_idx < len(old_lines):
        reconstructed_lines.append(old_lines[old_line_idx])
        old_line_idx += 1
    
    return '\n'.join(reconstructed_lines)

def extract_key_identifiers(line):
    """Extract key identifiers from a line for matching purposes"""
    cleaned = re.sub(r'["\'\s=\(\)]', '', line.lower())
    identifiers = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', cleaned)
    return set(identifiers)

def are_lines_similar(line1, line2, threshold=0.6):
    """Check if two lines are similar based on common identifiers"""
    ids1 = extract_key_identifiers(line1)
    ids2 = extract_key_identifiers(line2)
    
    if not ids1 and not ids2:
        return True
    if not ids1 or not ids2:
        return False
    
    intersection = len(ids1.intersection(ids2))
    union = len(ids1.union(ids2))
    
    return intersection / union >= threshold

def inject_assertions(latest_nb, old_nb):
    """Improved version that maintains assertion positioning relative to code"""
    # Normalize notebooks
    try:
        if hasattr(nbformat, 'normalize'):
            latest_nb = nbformat.normalize(latest_nb)
            old_nb = nbformat.normalize(old_nb)
    except Exception as e:
        print(f"Warning: Could not normalize notebooks: {e}")
    
    # Get cells
    latest_cells = []
    if hasattr(latest_nb, 'cells'):
        latest_cells = latest_nb.cells
    elif isinstance(latest_nb, dict) and 'cells' in latest_nb:
        latest_cells = latest_nb['cells']
    
    old_cells = []
    if hasattr(old_nb, 'cells'):
        old_cells = old_nb.cells
    elif isinstance(old_nb, dict) and 'cells' in old_nb:
        old_cells = old_nb['cells']
    
    # Create new cells list
    modified_cells = []
    
    import_statement = "import nbtest"
    import_exists = False
    
    for cell in old_cells:
        cell_type = get_cell_type(cell)
        if cell_type == 'code':
            source = get_cell_source(cell)
            if import_statement in source:
                import_exists = True
                break
    
    if not import_exists:
        import_cell = nbformat.v4.new_code_cell(source=import_statement)
        modified_cells.append(import_cell)
    
    # Process each old cell
    for old_cell in old_cells:
        cell_type = get_cell_type(old_cell)
        
        if cell_type == 'code':
            old_source = get_cell_source(old_cell)
            main_old_code = extract_main_code(old_source)
            
            # Look for matching code in latest notebook
            match_found = False
            
            for latest_cell in latest_cells:
                latest_cell_type = get_cell_type(latest_cell)
                
                if latest_cell_type == 'code':
                    latest_source = get_cell_source(latest_cell)
                    main_latest_code = extract_main_code(latest_source)
                    
                    if is_code_match(main_old_code, main_latest_code):
                        match_found = True
                        
                        # Extract code blocks with their assertions from latest cell
                        latest_code_blocks = extract_code_with_assertions(latest_source)
                        
                        # Reconstruct old code with properly positioned assertions
                        updated_source = reconstruct_code_with_assertions(main_old_code, latest_code_blocks)
                        
                        # Create a new cell with the updated source
                        new_cell = nbformat.v4.new_code_cell(source=updated_source)
                        
                        # Copy metadata if available
                        if hasattr(old_cell, 'metadata'):
                            new_cell.metadata = old_cell.metadata
                        elif isinstance(old_cell, dict) and 'metadata' in old_cell:
                            new_cell.metadata = old_cell['metadata']
                        
                        modified_cells.append(new_cell)
                        break
            
            if not match_found:
                new_cell = nbformat.v4.new_code_cell(source=old_source)
                
                if hasattr(old_cell, 'metadata'):
                    new_cell.metadata = old_cell.metadata
                elif isinstance(old_cell, dict) and 'metadata' in old_cell:
                    new_cell.metadata = old_cell['metadata']
                
                modified_cells.append(new_cell)
        else:
            # For non-code cells, create a new cell of the appropriate type
            if cell_type == 'markdown':
                source = get_cell_source(old_cell)
                new_cell = nbformat.v4.new_markdown_cell(source=source)
            elif cell_type == 'raw':
                source = get_cell_source(old_cell)
                new_cell = nbformat.v4.new_raw_cell(source=source)
            else:
                # Default to a code cell for unknown types
                source = get_cell_source(old_cell)
                new_cell = nbformat.v4.new_code_cell(source=source)
            
            # Copy metadata if available
            if hasattr(old_cell, 'metadata'):
                new_cell.metadata = old_cell.metadata
            elif isinstance(old_cell, dict) and 'metadata' in old_cell:
                new_cell.metadata = old_cell['metadata']
            
            modified_cells.append(new_cell)
    
    # Update the cells in the output notebook
    if hasattr(old_nb, 'cells'):
        old_nb.cells = modified_cells
    elif isinstance(old_nb, dict) and 'cells' in old_nb:
        old_nb['cells'] = modified_cells
    else:
        old_nb = nbformat.v4.new_notebook(cells=modified_cells)
        
        # Copy metadata if available
        if hasattr(old_nb, 'metadata'):
            old_nb.metadata = old_nb.metadata
        elif isinstance(old_nb, dict) and 'metadata' in old_nb:
            old_nb.metadata = old_nb['metadata']
    
    return old_nb

def run_tests(notebook_path):
    """Run pytest on the notebook and return a simple pass/fail result with any failure info"""
    try:
        command = f"pytest --nbtest -v {notebook_path}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # Simple check if all tests passed
        all_tests_passed = result.returncode == 0
        
        # Extract basic test counts
        output_lines = result.stdout.splitlines()
        test_counts = get_assertion_count(output_lines)
        print(f"Test counts: {test_counts}")
        if all_tests_passed:
            return True, test_counts, output_lines
        else:
            return False, test_counts, output_lines
            
    except Exception as e:
        print(f"Error running tests on {notebook_path}: {str(e)}")
        return False, [{"error": str(e)}], {"total": 0, "passed": 0, "failed": 0}, []
    
def extract_assertion_ids(output_lines):
    """Extract assertion IDs from pytest output"""
    failed_ids = []
    passed_ids = []

    for line in output_lines:
        # Look for patterns like "notebook.ipynb::4 FAILED" or "notebook.ipynb::3 PASSED"
        match = re.search(r'::(\d+)\s+(FAILED|PASSED)', line)
        if match:
            assertion_id = int(match.group(1))
            status = match.group(2)
            if status == "FAILED":
                failed_ids.append(assertion_id)
            elif status == "PASSED":
                passed_ids.append(assertion_id)
    
    return failed_ids, passed_ids

def get_assertion_count(output_lines):
    test_counts = {
        'total': 0,
        'passed': 0,
        'failed': 0
    }
    
    for line in output_lines:
        # Check for the combined passed/failed format
        combined_match = re.search(r'(\d+)\s+failed,\s+(\d+)\s+passed\s+in', line)
        if combined_match:
            failed_count = int(combined_match.group(1))
            passed_count = int(combined_match.group(2))
            test_counts['failed'] = failed_count
            test_counts['passed'] = passed_count
            test_counts['total'] = failed_count + passed_count
            continue
            
        # Check for only failed tests
        failed_match = re.search(r'(\d+)\s+failed\s+in', line)
        if failed_match and 'passed' not in line:  # Ensure we don't double count
            failed_count = int(failed_match.group(1))
            test_counts['failed'] = failed_count
            test_counts['total'] += failed_count
            continue
            
        # Check for only passed tests
        passed_match = re.search(r'(\d+)\s+passed\s+in', line)
        if passed_match and 'failed' not in line:  # Ensure we don't double count
            passed_count = int(passed_match.group(1))
            test_counts['passed'] = passed_count
            test_counts['total'] += passed_count
            continue
    
    return test_counts

class VersionTester:
    def __init__(self, kaggle_versions_folder, notebook, assertions_csv, versions_csv, datasets_csv=None):
        self.kaggle_versions_folder = kaggle_versions_folder
        self.notebook_path = notebook
        if datasets_csv:
            self.datasets_df = pd.read_csv(datasets_csv)
        else:
            self.datasets_df = None
        # Load assertion types from CSV
        self.assertions_df = pd.read_csv(assertions_csv)
        self.urls_df = pd.read_csv(versions_csv)
        self.results = []
        self.test_stats = []
        self.killed_versions = []  # List to track killed versions
        
        proj_name = os.path.basename(assertions_csv).replace("_assertions.csv", "")
        self.result_dir = os.path.dirname(assertions_csv)
        
        self.version_result_dir = os.path.join(self.result_dir, f"version_results")
        os.makedirs(self.version_result_dir, exist_ok=True)
        log_file_path = os.path.join(self.version_result_dir, "versionEval.log")
        self.logger = setup_logger(log_file_path, "versionEval")

        # Data structure for JSON format
        self.notebook_results = {}
        
        # Data structure for new CSV format
        self.detailed_results = []
    
    def get_assertion_type_by_id(self, assertion_id):
        """Get assertion type by ID from assertions CSV"""
        matches = self.assertions_df[self.assertions_df['Assertion_id'] == assertion_id]
        if not matches.empty:
            return matches.iloc[0]['Assertion_type']
        return "UNKNOWN"
    
    def categorize_assertion(self, assertion_type):
        """Categorize the assertion into dataset, model_perf, or model_arch"""
        if assertion_type.startswith("DATASET"):
            return "dataset"
        elif assertion_type.startswith("MODEL_PERF"):
            return "model_perf"
        elif assertion_type.startswith("MODEL_ARCH"):
            return "model_arch"
        return "other"

    def start_process(self):
        # notebook_filename = os.path.basename(self.notebook_path)
        # print(f"Processing notebook: {notebook_filename}")
        
        folder_path = self.kaggle_versions_folder
        # print(f"Processing versions in folder: {folder_path} with asserted notebook: {self.notebook_path}")

        version_files = [f for f in os.listdir(folder_path) if f.endswith('.ipynb')]
        self.logger.info(f"Found {len(version_files)} notebook versions in {folder_path}")

        if not version_files:
            print(f"No notebook versions found in {folder_path}")
            return
            
        base_name = self.extract_base_notebook_name(version_files)

        self.process_version_folder(folder_path, base_name, self.notebook_path)
        
        # Export results in the new CSV format
        self.export_detailed_results_csv()

    def extract_base_notebook_name(self, version_files):
        """Extract the base notebook name from version filenames"""
        if not version_files:
            return "unknown_notebook"
            
        sample_name = version_files[0]
        base_name = re.sub(r'_version_\d+|_v\d+', '', sample_name)
        base_name = os.path.splitext(base_name)[0]
        return base_name

    def execute_notebook(self, input_path, output_path):
        exe_error = False
        self.logger.info(f"Executing notebook: {input_path}")
        try:
            result = subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    input_path,
                    "--output",
                    output_path
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            if "Traceback" in result.stderr:
                exe_error = True
                self.logger.warning(f"Traceback detected during notebook execution of {input_path}")

            self.logger.debug(f"Execution stdout:\n{result.stdout}")
            self.logger.debug(f"Execution stderr:\n{result.stderr}")

        except subprocess.CalledProcessError as e:
            exe_error = True
            self.logger.error(f"Subprocess failed for {input_path}", exc_info=True)
            self.logger.debug(f"Execution stdout:\n{e.stdout}")
            self.logger.debug(f"Execution stderr:\n{e.stderr}")
        except Exception as e:
            exe_error = True
            self.logger.error(f"Unexpected error when executing {input_path}", exc_info=True)

        return exe_error


    def process_version_folder(self, folder_path, base_notebook_name, asserted_notebook_path):
        
        asserted_nb = nbformat.read(asserted_notebook_path, as_version=4)
        
        notebook_name = os.path.basename(asserted_notebook_path)
        if not notebook_name:
            notebook_name = "notebook"
        
        base_notebook_url = f"https://www.kaggle.com/{base_notebook_name}"

        if notebook_name not in self.notebook_results:
            self.notebook_results[notebook_name] = {}
        
        for notebook in sorted(os.listdir(folder_path)):
            if not notebook.endswith('.ipynb'):
                continue
            if not is_valid_notebook(os.path.join(folder_path, notebook)):
                self.logger.error(f"Invalid notebook: {notebook}")
                continue

            version_path = os.path.join(folder_path, notebook)
            self.logger.info(f"Testing version: {notebook}")
            
            version_id = self.extract_version_id(notebook)
            version_dir = os.path.join(self.version_result_dir, f"version_{version_id}")
            os.makedirs(version_dir, exist_ok=True)

            try:
                matching_versions = self.urls_df[
                    (self.urls_df['notebook_link'].str.contains(base_notebook_name, na=False)) & 
                    (self.urls_df['version_number'] == int(version_id.split('_')[-1]))
                ]
                if not matching_versions.empty:
                    version_url = matching_versions.iloc[0]['version_link']
                else:
                    # Fallback to a constructed URL if not found
                    version_url = f"{base_notebook_url}/version/{version_id.split('_')[-1]}"
                    self.logger.info(f"Warning: No matching version URL found for {notebook}, using fallback URL")
            except Exception as e:
                print(f"Error extracting version URL for {notebook}: {str(e)}")
                version_url = f"{base_notebook_url}/version/{version_id.split('_')[-1]}"
            
            if self.datasets_df:
                self.update_dataset_paths(version_path)
            
            # Initialize the result record
            result_record = {
                'notebook_url': version_url,
                'version_id': version_id,
                'original_notebook_url': base_notebook_url,
                'execute_no_error': 0,  # Default to error
                'num_dataset_assert_killed': 0,
                'num_model_perf_assert_killed': 0,
                'num_model_arch_assert_killed': 0,
                'total_asserts_killed': 0
            }

            
            version_with_assertions = self.inject_assertions_to_version(version_path, asserted_nb)
    

            if not os.path.exists(version_with_assertions):
                self.logger.error(f"Failed to inject assertions into {version_path}")
                self.detailed_results.append(result_record)
                continue

    
            output_path = os.path.join(self.version_result_dir, f"version_{version_id}_output.ipynb")
            exe_error = self.execute_notebook(version_with_assertions, output_path)
    
            if exe_error:
                self.detailed_results.append(result_record)
                continue
            else:
                result_record['execute_no_error'] = 1
            
            version_pytest_dir = os.path.join(self.version_result_dir, f"pytest_res")
            os.makedirs(version_pytest_dir, exist_ok=True)

            notebook_fname = (Path(version_with_assertions).stem).split("_chebyshev")[0]
            run_missing_tests(version_pytest_dir, version_with_assertions, 1, version_pytest_dir, mutant_type=notebook_fname)
            notebook_name = notebook.replace(".ipynb", "")
            pytest_csv_path = os.path.join(version_pytest_dir, f"{notebook_name}_with_assertions_pytest_testid_res_1.csv")
            # self.logger.info(f"pytest_csv_path: {pytest_csv_path}")
            
            if os.path.exists(pytest_csv_path):
                pytest_df = pd.read_csv(pytest_csv_path, header=None)
                for index, row in pytest_df.iterrows():
                    nbtest_id = row.iloc[0] 
                    result = row.iloc[1]
                    if result == 0:
                        assertion_type = self.get_assertion_type_by_id(nbtest_id)
                        if assertion_type.startswith("DATASET"):
                            result_record['num_dataset_assert_killed'] += 1
                        elif assertion_type.startswith("MODEL_PERF"):
                            result_record['num_model_perf_assert_killed'] += 1
                        elif assertion_type.startswith("MODEL_ARCH"):
                            result_record['num_model_arch_assert_killed'] += 1
                        result_record['total_asserts_killed'] += 1
                self.detailed_results.append(result_record)
                # self.logger.info(f"result_record: {result_record}")
                   
         

    def validate_asserted_notebook(self, asserted_notebook_path):
        """Validate that the asserted notebook passes all its tests"""
        print(f"Validating that asserted notebook passes all tests: {asserted_notebook_path}")
        all_tests_pass, test_counts, output_lines = run_tests(asserted_notebook_path)
        
        if all_tests_pass:
            # print(f"✅ Asserted notebook PASSED all tests ({test_counts['total']} tests)")
            return True
        else:
            # print(f"❌ Asserted notebook FAILED {test_counts['failed']} of {test_counts['total']} tests")
            return False

    def extract_version_id(self, notebook_filename):
        """Extract version ID from notebook filename"""
        # Try to extract a version number from the filename
        version_match = re.search(r'version(\d+)|v(\d+)', notebook_filename, re.IGNORECASE)
        if version_match:
            # Use the first matching group that's not None
            for group in version_match.groups():
                if group is not None:
                    return f"version_{group}"
        
        # If no version number found, use the filename without extension
        return os.path.splitext(notebook_filename)[0]

    def update_dataset_paths(self, notebook_path):
        notebook = nbformat.read(notebook_path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.source = self.replace_dataset_paths(cell.source)
        nbformat.write(notebook, notebook_path)


    def replace_dataset_paths(self, code):
        for _, row in self.datasets_df.iterrows():
            old_path_prefix = f"/kaggle/input/{row['dataset_name']}/"
            filename = row['filename']
            old_path = os.path.join(old_path_prefix, filename).replace("\\", "/")
            new_path = row['local_path'].replace("\\", "/")
            code = code.replace(old_path, new_path)
        return code

    def inject_assertions_to_version(self, version_path, asserted_nb):
        """Inject assertions from asserted notebook into a version"""
        try:
            old_notebook = nbformat.read(version_path, as_version=4)
            
            # Try to normalize the notebook
            if hasattr(nbformat, 'normalize'):
                old_notebook = nbformat.normalize(old_notebook)
            
            # Inject assertions
            modified_notebook = inject_assertions(asserted_nb, old_notebook)
            
            if modified_notebook is None:
                return None
            
            nb_version_name = os.path.basename(version_path)
            nb_version_name_assert = nb_version_name.replace(".ipynb", "_with_assertions.ipynb")
            temp_path = os.path.join(self.version_result_dir, nb_version_name_assert)
            nbformat.write(modified_notebook, temp_path)
            self.logger.info(f"Wrote notebook with assertions to {temp_path}")
            
            return temp_path
            
        except Exception as e:
            # add traceback 
            self.logger.error("Error when injecting assertions to version:\n%s", traceback.format_exc())
            return None


    def export_detailed_results_csv(self):
        """Export detailed testing results in the new CSV format"""
        df = pd.DataFrame(self.detailed_results)

        
        version_mutation_results_csv = os.path.join(self.version_result_dir, "version_mutation_results.csv")
        df.to_csv(version_mutation_results_csv, index=False)
        print(f"Exported detailed results to {version_mutation_results_csv}")
        print(f"Results contain {len(df)} notebook versions with assertion details")

def main():
    parser = argparse.ArgumentParser(description='Test Kaggle notebook versions with assertions')
    parser.add_argument('-k', '--kaggle_versions_folder', required=True,
                       help='Folder containing version folders')
    parser.add_argument('-n', '--notebook', required=True,
                       help='Path to the asserted notebook')
    parser.add_argument('-d', '--datasets_csv', default=None,
                        help='Optional: CSV file with dataset paths')
    parser.add_argument('-a', '--assertions_csv', required=True,
                       help='CSV file with assertion types')
    parser.add_argument('-v', '--version_csv', required=True,
                       help='CSV file with notebook versions')
    
    args = parser.parse_args()

    if args.datasets_csv:
        datasets_csv = args.datasets_csv
    else:
        datasets_csv = None
    
    tester = VersionTester(
        args.kaggle_versions_folder,
        args.notebook,
        args.assertions_csv,
        args.version_csv,
        args.datasets_csv
    )
    
    tester.start_process()

if __name__ == "__main__":
    main()