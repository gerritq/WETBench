#!/bin/bash
#SBATCH --job-name=p-qafe-viX
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --partition=nmes_gpu,gpu            
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00

nvidia-smi

module load anaconda3
source activate qa_env
python --version

LANG="vi"
SUBSET="extend"
DS="paras"
IN_FILE="/scratch_tmp/users/k21157437/paras/data/${LANG}/ds/eval/${LANG}_paras_${SUBSET}.jsonl"
OUT_FILE="/scratch_tmp/users/k21157437/paras/data/${LANG}/metrics/${LANG}_paras_${SUBSET}.jsonl"
#PROMPT_TECHNIQUES=("minimal" "cp" "rag")
PROMPT_TECHNIQUES=("cp" "rag")

python /scratch_tmp/users/k21157437/aid/scorers/qafe.py \
    --lang $LANG \
    --ds $DS \
    --in_file $IN_FILE \
    --out_file $OUT_FILE \
    --prompt_techs "${PROMPT_TECHNIQUES[@]}" 
