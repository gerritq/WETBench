#!/bin/bash
#SBATCH --job-name=s-brb-pt
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --partition=nmes_cpu,cpu
#SBATCH --time=01:30:00

source /scratch_tmp/users/k21157437/aid_env/bin/activate
which python 

LANG="pt"
IN_FILE="/scratch_tmp/users/k21157437/sums/data/${LANG}/eval/${LANG}_sums_eval.jsonl"
OUT_FILE="/scratch_tmp/users/k21157437/sums/data/${LANG}/eval/metrics/${LANG}_sums_eval.jsonl"
PROMPT_TECHNIQUES=("minimal" "instruct" "few1" "few2" "few3") # "instruct" "few1" "few2" "few3
TOTAL_N=270

python /scratch_tmp/users/k21157437/aid/scorers/brb.py \
    --lang $LANG \
    --in_file $IN_FILE \
    --out_file $OUT_FILE \
    --prompt_techs "${PROMPT_TECHNIQUES[@]}" \
    --total_n $TOTAL_N
