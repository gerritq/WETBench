#!/bin/bash
#SBATCH --job-name=nm-vi-qw
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

export HF_HOME="/scratch_tmp/prj/inf_nlg_ai_detection/.cache"

#Job takes ~3:40 
echo "Job started at: $(date)"

nvidia-smi

source /scratch_tmp/users/k21157437/aid_env/bin/activate
which python 

TASK="tst"
TOTAL_N=2700
LANG='vi'
SET="mpairs"
N_SHOTS=5
BATCH_SIZE=8

MODEL_NAME='Qwen/Qwen2.5-7B-Instruct' # "mistralai/Mistral-7B-Instruct-v0.3"
#MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" # "mistralai/Mistral-7B-Instruct-v0.3"

IN_FILE="/scratch_tmp/users/k21157437/neutral_new/data/${LANG}/datasets/4_${LANG}_${SET}.jsonl"

if [[ "$MODEL_NAME" == "Qwen/Qwen2.5-7B-Instruct" ]]; then
  MODEL_ABB="qwen"
elif [[ "$MODEL_NAME" == "mistralai/Mistral-7B-Instruct-v0.3" ]]; then
  MODEL_ABB="mistral"
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


python /scratch_tmp/users/k21157437/aid/mgt/mgt_hf.py \
  --total_n $TOTAL_N \
  --lang $LANG \
  --task $TASK \
  --in_file $IN_FILE \
  --out_file $OUT_FILE \
  --model_name $MODEL_NAME \
  --prompt_template_file $PROMPT_TEMPLATE_FILE \
  --few_shots_file $FS_FILE \
  --n_shots $N_SHOTS \
  --batch_size $BATCH_SIZE

echo "Job finished at: $(date)"