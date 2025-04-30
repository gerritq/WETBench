#!/bin/bash
#SBATCH --job-name=s-mgt-ev-vi
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --partition=nmes_cpu,cpu
#SBATCH --time=01:00:00

source /scratch_tmp/users/k21157437/aid_env/bin/activate
which python 

LANG="vi"
DS="sums"
N=270
IN_FILE="/scratch_tmp/users/k21157437/sums/data/${LANG}/ds/${LANG}_sums_eval.jsonl"
OUT_FILE="/scratch_tmp/users/k21157437/sums/data/${LANG}/eval/${LANG}_sums_eval.jsonl"
PROMPT_DIR="/scratch_tmp/users/k21157437/sums/code/prompts/${LANG}"
FS_FILE="/scratch_tmp/users/k21157437/sums/code/prompts/${LANG}/shots_${LANG}.jsonl"
PROMPT_TECHNIQUES=("minimal" "instruct" "few1" "few2" "few3") # "instruct" "few1" "few2" "few3"

python /scratch_tmp/users/k21157437/aid/mgt/mgt.py \
    --lang $LANG \
    --ds $DS \
    --in_file $IN_FILE \
    --out_file $OUT_FILE \
    --prompt_dir $PROMPT_DIR \
    --few_shots_file $FS_FILE \
    --prompt_techs "${PROMPT_TECHNIQUES[@]}"  \
    --n $N
