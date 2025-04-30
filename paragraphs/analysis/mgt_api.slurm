#!/bin/bash
#SBATCH --job-name=mgt-gpt-en
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --partition=nmes_cpu,cpu
#SBATCH --time=02:00:00
#SBATCH --mem=12G

export HF_HOME="/scratch/prj/inf_nlg_ai_detection/scratch_tmp/.cache"

echo "Job started at: $(date)"

source /scratch/users/k21157437/aid_venv/bin/activate
which python

TOTAL_N=2700
LANG='en'
TASK="extend"
MODEL_NAME='gpt-4o-mini'
#MODEL_NAME='gemini/gemini-2.0-flash'

CODE_DIR="/scratch/users/k21157437/paras/code"
DATA_DIR="/scratch/users/k21157437/paras/data"
IN_FILE="${DATA_DIR}/${LANG}/ds/${LANG}_paras_context_${TASK}.jsonl"
PROMPT_TEMPLATE_FILE="${CODE_DIR}/prompts/${LANG}/${TASK}_rag_${LANG}.txt"

#MODEL_NAME='gemini/gemini-2.0-flash'
if [[ "$MODEL_NAME" == "gpt-4o-mini" ]]; then
  MODEL_ABB="gpt"
elif [[ "$MODEL_NAME" == "gemini/gemini-2.0-flash" ]]; then
  MODEL_ABB="gemini"
fi

OUT_FILE="${DATA_DIR}/${LANG}/ds/mgt/${LANG}_paras_rag_${TASK}_${MODEL_ABB}.jsonl"

python /scratch/users/k21157437/aid/mgt/mgt_api.py \
  --total_n $TOTAL_N \
  --lang $LANG \
  --task $TASK \
  --in_file $IN_FILE \
  --out_file $OUT_FILE \
  --model_name $MODEL_NAME \
  --prompt_template_file $PROMPT_TEMPLATE_FILE

echo "Job finished at: $(date)"