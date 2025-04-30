#!/bin/bash
#SBATCH --job-name=brb-viX
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --partition=nmes_cpu,cpu
#SBATCH --time=00:40:00

source /scratch_tmp/users/k21157437/aid_env/bin/activate
which python 

LANG="vi"
SUBSET="extend"
TOTAL_N=270
IN_FILE="/scratch_tmp/users/k21157437/paras/data/${LANG}/ds/eval/${LANG}_paras_${SUBSET}.jsonl"
OUT_FILE="/scratch_tmp/users/k21157437/paras/data/${LANG}/metrics/${LANG}_paras_${SUBSET}.jsonl"
PROMPT_TECHNIQUES=("minimal" "cp" "rag")

python /scratch_tmp/users/k21157437/aid/scorers/brb.py \
    --lang $LANG \
    --in_file $IN_FILE \
    --out_file $OUT_FILE \
    --prompt_techs "${PROMPT_TECHNIQUES[@]}" \
    --total_n $TOTAL_N 