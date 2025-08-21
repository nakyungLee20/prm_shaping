#!/bin/sh
#SBATCH -J bon_gsm_shap_1.5b
#SBATCH -p gpu04
#SBATCH --gres=gpu:4
#SBATCH --ntasks=12
#SBATCH -o /home/leena/prm_shaping/logs/%x_%A.out

RUN="python"
FILE="/home/leena/prm_shaping/inference/prm_gsm8k_eval_bon.py"

for se in 0
do
$RUN $FILE
done