#!/bin/bash
#SBATCH --job-name=mgt-vi-gpt
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --partition=nmes_cpu,cpu
#SBATCH --time=02:00:00
#SBATCH --mem=12G

export HF_HOME="/scratch_tmp/prj/inf_nlg_ai_detection/.cache"

echo "Job started at: $(date)"

source /scratch_tmp/users/k21157437/aid_env/bin/activate
which python 

TOTAL_N=2700
LANG='vi'
TASK="sums"
IN_FILE="/scratch_tmp/users/k21157437/sums/data/${LANG}/ds/${LANG}_sums.jsonl"
MODEL_NAME='gpt-4o-mini'
#MODEL_NAME='gemini/gemini-2.0-flash'
if [[ "$MODEL_NAME" == "gpt-4o-mini" ]]; then
  MODEL_ABB="gpt"
elif [[ "$MODEL_NAME" == "gemini/gemini-2.0-flash" ]]; then
  MODEL_ABB="gemini"
fi

OUT_FILE="/scratch_tmp/users/k21157437/sums/data/${LANG}/ds/${LANG}_sums_mgt_few1_${MODEL_ABB}.jsonl"
PROMPT_TEMPLATE_FILE="/scratch_tmp/users/k21157437/sums/code/prompts/${LANG}/few_${LANG}.txt"
FEW_SHOTS_FILE="/scratch_tmp/users/k21157437/sums/code/prompts/${LANG}/shots_${LANG}.jsonl"
N_SHOTS=1

python /scratch_tmp/users/k21157437/aid/mgt/mgt_api.py \
  --total_n $TOTAL_N \
  --lang $LANG \
  --task $TASK \
  --in_file $IN_FILE \
  --out_file $OUT_FILE \
  --model_name $MODEL_NAME \
  --prompt_template_file $PROMPT_TEMPLATE_FILE \
  --few_shots_file $FEW_SHOTS_FILE \
  --n_shots $N_SHOTS 

echo "Job finished at: $(date)"