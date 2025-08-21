#!/bin/sh
#SBATCH -J naive_math_eval
#SBATCH -p gpu03
#SBATCH --gres=gpu:4
#SBATCH --ntasks=12
#SBATCH -o /home/leena/prm_shaping/logs/%x_%A.out

RUN="python"
FILE="/home/leena/prm_shaping/inference/math_eval_vllm.py"

for se in 0
do
$RUN $FILE
done