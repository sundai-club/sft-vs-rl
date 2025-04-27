#!/bin/bash

# === Configuration ===
TOTAL_CHUNKS=10
# List the GPU IDs available for use (e.g., 0 1 2 3 if you have 4 GPUs)
GPUS_AVAILABLE=(0 1 2 3) 
# Slurm partition name (CHANGE THIS)
SLURM_PARTITION="your_gpu_partition" 
# Estimated time per job (CHANGE THIS if needed)
TIME_LIMIT="02:00:00" 
# Memory per job (CHANGE THIS if needed)
MEM_PER_JOB="32G" 
# CPUs per job (CHANGE THIS if needed)
CPUS_PER_TASK=4
# Path to the run_evaluation.sh script
EVAL_SCRIPT_PATH="./run_evaluation.sh" 
# Directory for Slurm output/error logs
SLURM_LOG_DIR="slurm_logs"
# === End Configuration ===

NUM_GPUS=${#GPUS_AVAILABLE[@]}

if [ ! -f "${EVAL_SCRIPT_PATH}" ]; then
    echo "Error: Evaluation script not found at ${EVAL_SCRIPT_PATH}"
    exit 1
fi

SCRIPT_ABS_PATH=$(realpath ${EVAL_SCRIPT_PATH})

mkdir -p "${SLURM_LOG_DIR}"

echo "Submitting ${TOTAL_CHUNKS} evaluation jobs..."

for (( i=0; i<${TOTAL_CHUNKS}; i++ ))
do
    # Assign GPU in a round-robin fashion
    GPU_IDX=$(( i % NUM_GPUS ))
    TARGET_GPU=${GPUS_AVAILABLE[GPU_IDX]}
    
    JOB_NAME="eval_chunk_${i}"
    OUTPUT_LOG="${SLURM_LOG_DIR}/${JOB_NAME}_%j.out" # %j is the Job ID
    ERROR_LOG="${SLURM_LOG_DIR}/${JOB_NAME}_%j.err"

    echo "Submitting job for chunk ${i} on GPU ${TARGET_GPU}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${OUTPUT_LOG}
#SBATCH --error=${ERROR_LOG}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --constraint="gpu_id:${TARGET_GPU}" # Request specific GPU ID (syntax might vary)
#SBATCH --nodes=1             # Run on a single node
#SBATCH --ntasks=1            # Run a single task
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM_PER_JOB}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE         # Don't export environment variables

echo "Starting job ${JOB_NAME} on $(hostname) at $(date)"
echo "Running on GPU: ${TARGET_GPU}"

# Run the modified evaluation script
# Pass Chunk Index, Total Chunks, and Target GPU ID
bash "${SCRIPT_ABS_PATH}" ${i} ${TOTAL_CHUNKS} ${TARGET_GPU}

JOB_EXIT_CODE=$?
echo "Job ${JOB_NAME} finished at $(date) with exit code \${JOB_EXIT_CODE}"
exit \${JOB_EXIT_CODE}

EOF

sleep 1 # Small delay between submissions
done

echo "All ${TOTAL_CHUNKS} jobs submitted."
echo "Monitor jobs using 'squeue -u \$USER'"
echo "Check logs in '${SLURM_LOG_DIR}' directory." 