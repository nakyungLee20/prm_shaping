#!/bin/sh
#SBATCH -J math_ev
#SBATCH -p gpu03
#SBATCH --gres=gpu:2
#SBATCH --ntasks=12
#SBATCH -o /home/leena/prm_shaping/logs/%x_%A.out

RUN="python"
FILE="/home/leena/prm_shaping/inference/math_eval_vllm2.py"

for se in 0
do
$RUN $FILE
done