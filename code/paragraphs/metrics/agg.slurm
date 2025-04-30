#!/bin/bash
#SBATCH --job-name=paras-agg
#SBATCH --output=../../logs/%A_%a.out
#SBATCH --error=../../logs/%A_%a.err
#SBATCH --partition=nmes_cpu,cpu
#SBATCH --time=00:10:00

source /scratch_tmp/users/k21157437/aid_env/bin/activate
which python 

LANG="en"
BASE_DIR="/scratch_tmp/users/k21157437/paras/data"
SUBSETS=("first" "extend")
PROMPT_TECHNIQUES=("minimal" "cp" "rag")

python /scratch_tmp/users/k21157437/aid/scorers/aggregate_paras.py \
    --base_dir $BASE_DIR \
    --prompt_techs "${PROMPT_TECHNIQUES[@]}" \
    --subsets "${SUBSETS[@]}"
