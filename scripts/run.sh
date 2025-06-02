#!/bin/bash
NBTEST_DIR="."
KAGGLE_EVAL_DIR="${NBTEST_DIR}/kaggle_versions_eval"

export NBTEST_DIR KAGGLE_EVAL_DIR

run_RQ1() {
    local nb_path=$1
    local dynamic_iteration=$2 # Number of iterations to generate assertions
    local stat_method=$3
    local conf_level=$4
    local pytest_ite=$5
    local mutest_ite=$6
    local result_dir=$7

    proj_name=$(basename "$nb_path" ".ipynb")

    notebook_w_assertions="${result_dir}/${proj_name}_${stat_method}_${conf_level}_${dynamic_iteration}.ipynb"

    ## RQ1
    local start_time=$(date +%s)

    python "${NBTEST_DIR}/main.py" -n "${dynamic_iteration}" -o "${result_dir}" -m "${stat_method}" -c "${conf_level}" "${nb_path}"
    echo python "${NBTEST_DIR}/main.py" -n "${dynamic_iteration}" -o "${result_dir}" -m "${stat_method}" -c "${conf_level}" "${nb_path}"

    local end_time=$(date +%s)
    local parse_time=$((end_time - start_time))
    local overhead_txt="${result_dir}/${proj_name}_overhead.txt"

    echo "${parse_time}" >> "${overhead_txt}"

    # RQ2. Run pytest
    python "${NBTEST_DIR}/main.py" --pytest -n "${dynamic_iteration}" -pn "${pytest_ite}"  -o "${result_dir}" -rundir "${run_dir}" "${notebook_w_assertions}"
    echo python "${NBTEST_DIR}/main.py" --pytest  -n "${dynamic_iteration}" -pn "${pytest_ite}"  -o "${result_dir}" -rundir "${run_dir}" "${notebook_w_assertions}"

    # RQ3. Run mutation testing
    if [[ -f "$notebook_w_assertions" ]]; then
        python "${NBTEST_DIR}/main.py" --mutest -o "${result_dir}" -mn "${mutest_ite}" -rundir "${run_dir}" "${notebook_w_assertions}"
        echo python "${NBTEST_DIR}/main.py" --mutest -o "${result_dir}" -mn "${mutest_ite}" -rundir "${run_dir}" "${notebook_w_assertions}"
    else
        echo "Not found: ${notebook_w_assertions}"
    fi

    if [[ -f "$notebook_w_assertions" ]]; then
        # RQ3.5: Download older Kaggle versions for evaluation
        local version_csv="${KAGGLE_EVAL_DIR}/filtered_notebook_versions.csv"
        local assertions_csv="${result_dir}/${proj_name}_assertions.csv"
        local other_versions_dir="${result_dir}/kaggle_other_versions"

        if [[ -d "${other_versions_dir}" ]]; then
            python "${KAGGLE_EVAL_DIR}/versionEval.py" -k "${other_versions_dir}" -n "${notebook_w_assertions}" -a "${assertions_csv}" -v "${version_csv}"
            echo "Command executed: python ${KAGGLE_EVAL_DIR}/versionEval.py -k ${other_versions_dir} -n ${notebook_w_assertions} -a ${assertions_csv} -v ${version_csv}"
        else
            echo "Not found: ${other_versions_dir}"
        fi

    else
        echo "Not found: ${notebook_w_assertions}"
    fi
}

export -f run_RQ1

run_RQ1 "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"
