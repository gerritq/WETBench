#!/bin/bash
#SBATCH --job-name=s-qafe-vi
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
IN_FILE="/scratch_tmp/users/k21157437/sums/data/${LANG}/eval/${LANG}_sums_eval.jsonl"
OUT_FILE="/scratch_tmp/users/k21157437/sums/data/${LANG}/eval/metrics/${LANG}_sums_eval.jsonl"
PROMPT_TECHNIQUES=("minimal" "instruct" "few1" "few2" "few3") #"instruct" "few1" "few2" "few3"

CUDA_LAUNCH_BLOCKING=1  python /scratch_tmp/users/k21157437/aid/scorers/qafe.py \
    --lang $LANG \
    --in_file $IN_FILE \
    --out_file $OUT_FILE \
    --prompt_techs "${PROMPT_TECHNIQUES[@]}"
