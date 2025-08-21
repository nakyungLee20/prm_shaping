#!/bin/sh
#SBATCH -J maj_gsm
#SBATCH -p gpu03
#SBATCH --gres=gpu:4
#SBATCH --ntasks=12
#SBATCH -o /home/leena/prm_shaping/logs/%x_%A.out

RUN="python"
FILE="/home/leena/prm_shaping/inference/prm_gsm8k_eval_bon.py"

for se in 0
do
$RUN $FILE
done