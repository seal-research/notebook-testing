import sys
import os
import json
import argparse
import pandas as pd
import time

proj_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(proj_folder, '../'))

from mutation_testing.gen_mutants import generate_mutants, load_notebook
from utils.pytest_run import run_missing_tests
from utils.utils import get_notebook_name, setup_logger, get_unfinished_mutant_idx

config_path = os.path.join(proj_folder, "../", "config.json")
global CONFIG_MUTANTS
with open(config_path, "r") as config_file:
    CONFIG_MUTANTS = json.load(config_file)["MUTANT_TYPES"]

MUTANT_TYPE_LIST = list(CONFIG_MUTANTS.keys())
MAX_MUTANT_COUNT = 4

def calculate_mutant_score(mutant_root_dir, notebook_path, pytest_iterations, rundir):
    notebook_fname = get_notebook_name(notebook_path)
    logs_dir = os.path.join(mutant_root_dir, f"A_logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logger = setup_logger(os.path.join(logs_dir, 'calculate_mutant_score.log'), 'calculate_mutant_score.log')

    # Test original notebook
    if not os.path.exists(os.path.join(mutant_root_dir, f"{notebook_fname}_pytest_testid_res_{1}.csv")):
        logger.debug(f"Running tests on original: {notebook_path}...")
        run_missing_tests(mutant_root_dir, notebook_path, 1, rundir)

    iter_txt_path = os.path.join(mutant_root_dir, f"{notebook_fname}_pytest_testid_res_{1}.csv")

    if os.path.exists(iter_txt_path):
        killed_mutants = []
        mutants_res_json = {}
        for mutant_type in MUTANT_TYPE_LIST:
            logger.info(f"Running mutation testing for {mutant_type}...")
            mutant_type_dir = os.path.join(mutant_root_dir, f"{mutant_type}")
            os.makedirs(mutant_type_dir, exist_ok=True)

            mutant_notebooks_dir = os.path.join(mutant_type_dir, "mutant_notebooks")
            unfinished_mutant_lst = get_unfinished_mutant_idx(mutant_notebooks_dir, MAX_MUTANT_COUNT, mutant_type, notebook_fname)
            mutant_files = []
            for i in range(MAX_MUTANT_COUNT):
                if str(i) not in unfinished_mutant_lst:
                    mutant_files.append(os.path.join(mutant_notebooks_dir, f"{notebook_fname}_mutant_{mutant_type}_{i}.ipynb"))

            asserts_df = pd.read_csv(os.path.join(mutant_root_dir, "..", f"{notebook_fname}_assertions.csv"))

            mutant_pytest_testid_res_dir = os.path.join(mutant_type_dir, "pytest_testid_res")
            os.makedirs(mutant_pytest_testid_res_dir, exist_ok=True)


            test_res = pd.read_csv(iter_txt_path, header=None, names=['testid', 'res'])
            prev_fail = test_res[test_res['res'] == 0]['testid'].tolist()

            # logger.debug(test_res['res'].tolist())


            # Run tests on mutant notebooks
            for mutant_path in mutant_files:
                if os.path.exists(mutant_path):

                    mutant_fname = os.path.basename(mutant_path).split('.ipynb')[0]
                    mutant_type_id = mutant_fname.split('_mutant_')[1]

                    mutant_type, mutant_id = mutant_type_id.rsplit('_', 1)

                    if not mutant_id.isdigit():
                        mutant_type, mutant_id = mutant_type_id, '0'

                    if mutant_type not in list(mutants_res_json.keys()):
                        mutants_res_json[mutant_type] = []

                    mutant_type_log_dir = os.path.join(mutant_type_dir, "logs")
                    res_dir = os.path.join(mutant_type_log_dir, f"{notebook_fname}_mutant_{mutant_type}_logs")
                    os.makedirs(res_dir, exist_ok=True)

                    pytest_logs_dir= os.path.join(mutant_type_dir, "pytest_nb_tracebacks_logs")
                    os.makedirs(pytest_logs_dir, exist_ok=True)
                    csv_dir= os.path.join(mutant_type_dir, "pytest_testid_res")
                    os.makedirs(csv_dir, exist_ok=True)

                    txt_dir= os.path.join(mutant_root_dir, "pytest_txt")
                    os.makedirs(txt_dir, exist_ok=True)
                    traceback_nb_dir = os.path.join(mutant_root_dir, "pytest_nb_tracebacks_notebooks")
                    os.makedirs(traceback_nb_dir, exist_ok=True)


                    run_missing_tests(traceback_nb_dir, mutant_path, pytest_iterations, rundir, mutant_type=mutant_fname, pytest_logs_dir=pytest_logs_dir, csv_dir=csv_dir, txt_dir=txt_dir)
                    logger.debug(f"Finished tests on mutant: {mutant_path}...")

                    for i in range(pytest_iterations):
                        logger.info(f"{mutant_fname}_{i+1}")
                        mut_iter_txt_path = os.path.join(csv_dir, f'{mutant_fname}_pytest_testid_res_{i+1}.csv')
                        try:
                            curr_res = pd.read_csv(mut_iter_txt_path, header=None, names=['testid', 'res'])
                        except FileNotFoundError:

                            logger.warning(f"{mut_iter_txt_path} not found. Attempting to re-run missing test...")

                            try:

                                run_missing_tests(traceback_nb_dir, mutant_path, pytest_iterations, rundir, mutant_type=mutant_fname, pytest_logs_dir=pytest_logs_dir, csv_dir=csv_dir, txt_dir=txt_dir)

                                curr_res = pd.read_csv(mut_iter_txt_path, header=None, names=['testid', 'res'])
                            except (FileNotFoundError, Exception) as e:
                                logger.warning(f"Failed to recover {mut_iter_txt_path}: {e}. Creating empty result.")


                                continue

                        curr_fail = curr_res[curr_res['res'] == 0]['testid'].tolist()
                        logger.debug(curr_res['res'].tolist())

                        diff_set = [i for i in curr_fail if i not in prev_fail]
                        diff_set = list(set(diff_set)) # Remove the duplicate items

                        if -1 in curr_res['res'].tolist():
                            killed_mutants.append([mutant_type, -1])
                        elif len(diff_set):
                            killed_mutants.append([mutant_type, 1])
                        else:
                            killed_mutants.append([mutant_type, 0])

                        mutant_csv_path = os.path.join(res_dir, 'mutation_performed.csv')
                        if (os.path.exists(mutant_csv_path)):
                            logger.debug(f"{mutant_type}, {mutant_id}")

                            if os.path.getsize(mutant_csv_path) > 0:
                                mutant_df_log = pd.read_csv(mutant_csv_path)
                                if not mutant_df_log.empty:
                                    mutant_cell, mutant_line = mutant_df_log['cell'].tolist()[0], mutant_df_log['line'].tolist()[0]
                                else:
                                    mutant_cell, mutant_line = 0, 0
                            else:
                                mutant_cell, mutant_line = 0, 0

                            after_mutant = []
                           
                            for id in curr_res['testid'].tolist():
                                if isinstance(id, str):
                                    test_id_split = id.split('_')
                                    if len(test_id_split) >= 2:
                                        cell, line = test_id_split[-2], test_id_split[-1]
                                        cell = int(cell)
                                        line = int(line)
                                        if (cell > mutant_cell) or (cell == mutant_cell and line > mutant_line):
                                            after_mutant.append(id)
                                    else:
                                        logger.debug(f"Warning: Unexpected test_id format: {id}")

                            killed_after = list(set([i for i in after_mutant if i in diff_set]))
                            not_killed_after = list(set([i for i in after_mutant if i not in killed_after]))

                            diff_set_asserts = asserts_df[asserts_df["Assertion_id"].isin(diff_set)]["Assertion_type"].tolist()
                            killed_after_asserts = asserts_df[asserts_df["Assertion_id"].isin(killed_after)]["Assertion_type"].tolist()
                            not_killed_after_asserts = asserts_df[asserts_df["Assertion_id"].isin(not_killed_after)]["Assertion_type"].tolist()

                            logger.debug(f"Diff_set: {str(diff_set)}")
                            logger.debug(f"Diff_set_type: {str(diff_set_asserts)}")
                            killed_df = pd.DataFrame({'testid': diff_set, 'assertion_type': diff_set_asserts})

                            killed_after_df = pd.DataFrame({'testid': killed_after, 'assertion_type': killed_after_asserts})

                            not_killed_after_df = pd.DataFrame({'testid': not_killed_after, 'assertion_type': not_killed_after_asserts})

                            if -1 not in curr_res['res'].tolist():
                                mutants_res_json[mutant_type].append({f"mutant_{mutant_id}": [{i: j} for i, j in zip(diff_set, diff_set_asserts)]})
                            else:
                                first_fail = curr_res[curr_res['res'] == -1]['testid'].tolist()[0]
                                fail_split = first_fail.split('_')
                                fail_cell, fail_line = int(fail_split[-2]), int(fail_split[-1])
                                killed_diff = []
                                if len(diff_set):
                                    for assert_id, assert_type in zip(diff_set, diff_set_asserts):
                                        test_id_split = assert_id.split('_')
                                        cell, line = test_id_split[-2], test_id_split[-1]
                                        cell = int(cell)
                                        line = int(line)
                                        if (cell < fail_cell) or (cell == mutant_cell and line < fail_line):
                                            killed_diff.append({assert_id: assert_type})
                                        else:
                                            break

                                    mutants_res_json[mutant_type].append({f"mutant_{mutant_id}": killed_diff})

                                mutants_res_json[mutant_type].append({f"mutant_{mutant_id}": [{"EXEC_ERROR": "EXEC_ERROR"}]})


        overall_res = pd.DataFrame(sorted(killed_mutants, key=lambda l: l[0]), columns=['mutant', 'result'])
        overall_res.to_csv(os.path.join(mutant_root_dir, 'mutation_testing_result.csv'), index=False)

        with open(os.path.join(mutant_root_dir, 'mutation_testing.json'), 'w') as f:
            json.dump(mutants_res_json, f)

    else:
        logger.debug(f"{iter_txt_path} doesn't exist. Stop computing mutant score.")

def mutation_testing(notebook_path, mutant_dir, pytest_iterations, rundir):
    nb = load_notebook(notebook_path)
    notebook_fname = get_notebook_name(notebook_path)
    generate_mutants(notebook_path, mutant_dir, notebook_fname)
    calculate_mutant_score(mutant_dir, notebook_path, pytest_iterations, rundir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mutation testing')
    parser.add_argument('notebook', type=str, help='The path to the Jupyter notebook file')
    parser.add_argument('-o', required=True, help='Absolute path for mutants directory')
    parser.add_argument('-pn', default="1", help='Number of iterations for running tests on each mutant')
    parser.add_argument('-rundir', default="./", help='Folder to run the notebook. This folder should have the file structure required for the notebook')

    args = parser.parse_args()
    mutant_dir = os.path.abspath(args.o)

    pytest_iterations = int(args.pn)
    notebook_path = os.path.abspath(args.notebook)

    # mutation_testing(notebook_path, mutant_dir, pytest_iterations, args.rundir)
    calculate_mutant_score(mutant_dir, notebook_path, pytest_iterations, args.rundir)
