#!/bin/bash
#SBATCH --job-name=mgt-vi-mis
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

# SBATCH --constraint=a100_80g

export HF_HOME="/scratch/prj/inf_nlg_ai_detection/scratch_tmp/.cache"

#Job takes ~3:40 
echo "Job started at: $(date)"

nvidia-smi

#source /scratch_tmp/users/k21157437/aid_env/bin/activate
source /scratch/users/k21157437/aid_venv/bin/activate
which python 

TOTAL_N=2700
LANG='vi'
TASK="sums"
IN_FILE="/scratch/users/k21157437/sums/data/${LANG}/ds/${LANG}_sums.jsonl"
#MODEL_NAME='Qwen/Qwen2.5-7B-Instruct' # "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" # "mistralai/Mistral-7B-Instruct-v0.3"
if [[ "$MODEL_NAME" == "Qwen/Qwen2.5-7B-Instruct" ]]; then
  MODEL_ABB="qwen"
elif [[ "$MODEL_NAME" == "mistralai/Mistral-7B-Instruct-v0.3" ]]; then
  MODEL_ABB="mistral"
fi

N_SHOTS=1
OUT_FILE="/scratch/users/k21157437/sums/data/${LANG}/ds/${LANG}_sums_mgt_few${N_SHOTS}_${MODEL_ABB}.jsonl"
PROMPT_TEMPLATE_FILE="/scratch/users/k21157437/sums/code/prompts/${LANG}/few_${LANG}.txt"
FEW_SHOTS_FILE="/scratch/users/k21157437/sums/code/prompts/${LANG}/shots_${LANG}.jsonl"
BATCH_SIZE=1

python /scratch/users/k21157437/aid/mgt/mgt_hf.py \
  --total_n $TOTAL_N \
  --lang $LANG \
  --task $TASK \
  --in_file $IN_FILE \
  --out_file $OUT_FILE \
  --model_name $MODEL_NAME \
  --prompt_template_file $PROMPT_TEMPLATE_FILE \
  --few_shots_file $FEW_SHOTS_FILE \
  --n_shots $N_SHOTS \
  --batch_size $BATCH_SIZE

echo "Job finished at: $(date)"