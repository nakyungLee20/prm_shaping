#!/bin/sh
#SBATCH -J scr_omni
#SBATCH -p gpu04
#SBATCH --array=0-3%2
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH -o logs/%x_%A_%a.out

set -euo pipefail

RUN="python"
FILE="/home/leena/prm/inference/eval_vllm3.py"

MODES=(ori cmi contri)
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"

echo "[$(date +'%F %T')] ArrayID=${SLURM_ARRAY_TASK_ID} MODE=${MODE}"
srun ${RUN} "${FILE}" --reward_type "${MODE}"
echo "[$(date +'%F %T')] Done MODE=${MODE}"