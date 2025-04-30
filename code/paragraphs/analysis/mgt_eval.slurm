#!/bin/bash
#SBATCH --job-name=p-mgt-viX
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --partition=nmes_cpu,cpu
#SBATCH --time=00:45:00

source /scratch_tmp/users/k21157437/aid_env/bin/activate
which python 

LANG="vi"
DS="paras"
SUBSET="extend"
N=270

IN_FILE="/scratch_tmp/users/k21157437/paras/data/${LANG}/ds/${LANG}_paras_context_${SUBSET}.jsonl"
OUT_FILE="/scratch_tmp/users/k21157437/paras/data/${LANG}/ds/eval/${LANG}_paras_${SUBSET}.jsonl"
PROMPT_DIR="/scratch_tmp/users/k21157437/paras/code/prompts/${LANG}"
PROMPT_TECHNIQUES=("minimal" "cp" "rag")

python /scratch_tmp/users/k21157437/aid/mgt/mgt.py \
    --lang $LANG \
    --ds $DS \
    --subset $SUBSET \
    --in_file $IN_FILE \
    --out_file $OUT_FILE \
    --prompt_dir $PROMPT_DIR \
    --prompt_techs "${PROMPT_TECHNIQUES[@]}" \
    --n $N