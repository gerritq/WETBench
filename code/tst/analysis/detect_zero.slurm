#!/bin/bash
#SBATCH --job-name=n0-enp
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --partition=rental_gpu
# SBATCH --partition=nmes_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=32G

export HF_HOME="/scratch/prj/inf_nlg_ai_detection/scratch_tmp/.cache"

nvidia-smi

source /scratch/users/k21157437/aid_venv/bin/activate
which python

LANGUAGES=("en_paras") 
MODELS=("gemini" "qwen" "mistral") # "gpt" 
TASK="neutral"
N_SHOTS=5
DETECTORS=("revise" "gecscore") #"revise" "gecscore"


for LANG in "${LANGUAGES[@]}"; do
    for MODEL in "${MODELS[@]}"; do

        echo "Processing LANG=${LANG}, TASK=${TASK}, MODEL=${MODEL} at $(date)"


        if [ "$LANG" = "en" ]; then
            IN_FILE="/scratch/users/k21157437/neutral_new/data/${LANG}/datasets/mgt/${LANG}_default_mgt_few${N_SHOTS}_${MODEL}.jsonl"
            OUT_FILE="/scratch/users/k21157437/neutral_new/data/${LANG}/detect/${LANG}_default_mgt_few${N_SHOTS}_${MODEL}_zero.jsonl"
        elif [ "$LANG" = "en_paras" ]; then
            IN_FILE="/scratch/users/k21157437/neutral_new/data/${LANG:0:2}/datasets/mgt/${LANG}_mgt_few${N_SHOTS}_${MODEL}.jsonl"
            OUT_FILE="/scratch/users/k21157437/neutral_new/data/${LANG:0:2}/detect/${LANG}_mgt_few${N_SHOTS}_${MODEL}_zero.jsonl"
        else
            IN_FILE="/scratch/users/k21157437/neutral_new/data/${LANG}/datasets/mgt/${LANG}_mgt_few${N_SHOTS}_${MODEL}.jsonl"
            OUT_FILE="/scratch/users/k21157437/neutral_new/data/${LANG}/detect/${LANG}_mgt_few${N_SHOTS}_${MODEL}_zero.jsonl"
        fi

        python /scratch/users/k21157437/aid/detectors/detect.py \
          --in_file "${IN_FILE}" \
          --out_file "${OUT_FILE}" \
          --task "${TASK}" \
          --lang "${LANG:0:2}" \
          --detectors "${DETECTORS[@]}"

  done
done
