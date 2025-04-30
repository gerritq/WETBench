#!/bin/bash
#SBATCH --job-name=na-enp-gem
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --partition=nmes_cpu,cpu
#SBATCH --time=02:00:00
#SBATCH --mem=12G

export HF_HOME="/scratch_tmp/prj/inf_nlg_ai_detection/.cache"

echo "Job started at: $(date)"

source /scratch_tmp/users/k21157437/aid_env/bin/activate
which python 

TASK="tst"
TOTAL_N=2700
LANG='en'
SET="paras"
N_SHOTS=5
#MODEL_NAME='gpt-4o-mini'
MODEL_NAME='gemini/gemini-2.0-flash'

IN_FILE="/scratch_tmp/users/k21157437/neutral_new/data/${LANG}/datasets/4_${LANG}_${SET}.jsonl"

if [[ "$MODEL_NAME" == "gpt-4o-mini" ]]; then
  MODEL_ABB="gpt"
elif [[ "$MODEL_NAME" == "gemini/gemini-2.0-flash" ]]; then
  MODEL_ABB="gemini"
fi


if [ "$LANG" = "en" ]; then
   OUT_FILE="/scratch_tmp/users/k21157437/neutral_new/data/${LANG}/datasets/mgt/${LANG}_${SET}_mgt_few${N_SHOTS}_${MODEL_ABB}.jsonl"
   PROMPT_TEMPLATE_FILE="/scratch_tmp/users/k21157437/neutral_new/code/prompts/${LANG}/few_${SET}_${LANG}.txt"
   FS_FILE="/scratch_tmp/users/k21157437/neutral_new/code/prompts/${LANG}/shots_${SET}_${LANG}.jsonl"
else
   OUT_FILE="/scratch_tmp/users/k21157437/neutral_new/data/${LANG}/datasets/mgt/${LANG}_mgt_few${N_SHOTS}_${MODEL_ABB}.jsonl"
   PROMPT_TEMPLATE_FILE="/scratch_tmp/users/k21157437/neutral_new/code/prompts/${LANG}/few_${LANG}.txt"
   FS_FILE="/scratch_tmp/users/k21157437/neutral_new/code/prompts/${LANG}/shots_${LANG}.jsonl"
fi
python /scratch_tmp/users/k21157437/aid/mgt/mgt_api.py \
  --total_n $TOTAL_N \
  --lang $LANG \
  --task $TASK \
  --in_file $IN_FILE \
  --out_file $OUT_FILE \
  --model_name $MODEL_NAME \
  --prompt_template_file $PROMPT_TEMPLATE_FILE \
  --few_shots_file $FS_FILE \
  --n_shots $N_SHOTS 

echo "Job finished at: $(date)"