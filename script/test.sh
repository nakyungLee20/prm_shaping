#!/bin/sh
#SBATCH -J tt_mmlu
#SBATCH -p gpu02
#SBATCH --gres=gpu:1
#SBATCH --ntasks=12
#SBATCH -o /home/leena/prm_shaping/logs/%x_%A.out

RUN="python"
FILE="/home/leena/prm_shaping/inference/mmlu_scorer.py"

for se in 0
do
$RUN $FILE
done