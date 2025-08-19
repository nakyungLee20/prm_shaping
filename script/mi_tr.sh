#!/bin/sh
#SBATCH -J gsm_incor
#SBATCH -p gpu04
#SBATCH --gres=gpu:1
#SBATCH --ntasks=12
#SBATCH -o /home/leena/prm_shaping/logs/%x_%A.out

RUN="python"
FILE="/home/leena/prm_shaping/data_generation/incorrect_step.py"

for se in 0
do
$RUN $FILE
done