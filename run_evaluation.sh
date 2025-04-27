#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-3B-Instruct" 
DATASET_PATH="/workspace/zero_zhiwei/sft-vs-rl/test.parquet"
K_VALUES=(1 4 16 64 256)          # pass@k values
TEMPERATURE=0.7
TOP_P=0.95
MAX_TOKENS=2048
TENSOR_PARALLEL_SIZE=1      
TARGET_GPU=3 
N_TEST=100     # Number of test samples to use
RANDOM_SEED=42 # Seed for sampling test samples

CONDA_BASE=$(conda info --base)
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "Conda initialization script not found. Please adjust the path in run_evaluation.sh" >&2
    exit 1
fi
conda activate zero
CONDA_ACTIVATE_EXIT_CODE=$?
if [ ${CONDA_ACTIVATE_EXIT_CODE} -ne 0 ]; then
    echo "Failed to activate conda environment 'zero'. Exit code: ${CONDA_ACTIVATE_EXIT_CODE}" >&2
    exit 1
fi
echo "Activated conda environment: zero"

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_pass_k.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_pass_k.py"
OUTPUT_DIR="${SCRIPT_DIR}/evaluation_output"
MODEL_NAME_SAFE=${MODEL_NAME//\//_} # Replace / with _ for filename
RESULTS_JSON="${OUTPUT_DIR}/${MODEL_NAME_SAFE}_results.json" 
LOG_FILE="${OUTPUT_DIR}/${MODEL_NAME_SAFE}_evaluation.log"
PASS_K_CSV="${OUTPUT_DIR}/${MODEL_NAME_SAFE}_pass_k.csv" 
PLOT_FILE="${OUTPUT_DIR}/${MODEL_NAME_SAFE}_pass_k_plot.png" 

mkdir -p "${OUTPUT_DIR}"
k_values_str=$(IFS=' '; echo "${K_VALUES[*]}")
 
n_test_arg=""
if [[ -n "${N_TEST}" ]]; then
  n_test_arg="--n_test ${N_TEST} --random_seed ${RANDOM_SEED}"
fi

echo "--- Running Evaluation --- "
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Targeting GPU: ${TARGET_GPU}"
echo "Saving detailed results to: ${RESULTS_JSON}"
echo "Saving pass@k results to: ${PASS_K_CSV}"
echo "Logging output to: ${LOG_FILE}"
echo "Number of test samples: ${N_TEST:-All}"

export CUDA_VISIBLE_DEVICES=${TARGET_GPU}

python "${EVAL_SCRIPT}" \
    --model_path "${MODEL_NAME}" \
    --dataset_path "${DATASET_PATH}" \
    --k_values ${k_values_str} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_tokens ${MAX_TOKENS} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --results_output_path "${RESULTS_JSON}" \
    --pass_k_output_path "${PASS_K_CSV}" \
    ${n_test_arg} \
    > "${LOG_FILE}" 2>&1

EVAL_EXIT_CODE=$?
unset CUDA_VISIBLE_DEVICES

if [ ${EVAL_EXIT_CODE} -ne 0 ]; then
    echo "Evaluation failed with exit code ${EVAL_EXIT_CODE}. Check log ${LOG_FILE} for details."
    exit ${EVAL_EXIT_CODE}
fi

echo "Evaluation completed successfully. Log saved to ${LOG_FILE}"

# --- Plotting Results ---
echo "\n--- Generating Plot --- "
if [ -f "${PASS_K_CSV}" ]; then
    python "${PLOT_SCRIPT}" \
        --input_csv "${PASS_K_CSV}" \
        --output_plot "${PLOT_FILE}" \
        --plot_title "(${MODEL_NAME})" \
        >> "${LOG_FILE}" 2>&1 # Append plotting output to the same log

    PLOT_EXIT_CODE=$?
    if [ ${PLOT_EXIT_CODE} -eq 0 ]; then
        echo "Plot generated successfully: ${PLOT_FILE}"
    else
        echo "Plot generation failed with exit code ${PLOT_EXIT_CODE}. Check log ${LOG_FILE} for details."
    fi
else
    echo "Pass@k CSV file not found (${PASS_K_CSV}), skipping plot generation."
fi

exit 0 # Exit with 0 if evaluation was successful, regardless of plotting outcome 