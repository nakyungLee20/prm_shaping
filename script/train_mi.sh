#!/bin/sh
#SBATCH -J mi_tr
#SBATCH -p gpu03
#SBATCH --gres=gpu:3
#SBATCH --ntasks=12
#SBATCH -o /home/leena/prm_shaping/logs/%x_%A.out

RUN="python"
FILE="/home/leena/prm_shaping/prm_training/training.py"

for se in 0
do
$RUN $FILE
done