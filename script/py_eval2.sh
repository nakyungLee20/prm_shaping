#!/bin/sh
#SBATCH -J olm_ev
#SBATCH -p gpu04
#SBATCH --gres=gpu:2
#SBATCH --ntasks=12
#SBATCH -o /home/leena/prm_shaping/logs/%x_%A.out

RUN="python"
FILE="/home/leena/prm_shaping/inference/olym_eval_vllm.py"

for se in 0
do
$RUN $FILE
done