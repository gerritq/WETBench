#!/bin/bash
#SBATCH --job-name=p0-viX-w
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
# SBATCH --partition=rental_gpu
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --begin=now+2hours

start_time=$(date +%s)
echo "Job started at: $(date)"

export HF_HOME="/scratch/prj/inf_nlg_ai_detection/scratch_tmp/.cache"

nvidia-smi

source /scratch/users/k21157437/aid_venv/bin/activate
which python

DATA_DIR="/scratch/users/k21157437/paras/data"
LANGS=("vi") #("en" "pt" 
TASKS=("first")
MODELS=("mistral") # "gpt" "gemini" "qwen" "mistral"
DETECTORS=("gecscore")
# "binoculars", "llr", "fastdetectgpt_white", "revise", "gecscore", "fastdetectgpt_black"

for LANG in "${LANGS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for MODEL in "${MODELS[@]}"; do

      echo " "
      echo "Running LANG=${LANG}, TASK=${TASK}, MODEL=${MODEL}"

      iter_start=$(date +%s)

      IN_FILE="${DATA_DIR}/${LANG}/ds/mgt/${LANG}_paras_rag_${TASK}_${MODEL}.jsonl"
      OUT_FILE="${DATA_DIR}/${LANG}/detect/${LANG}_paras_rag_${TASK}_${MODEL}_zero.jsonl"

      python /scratch/users/k21157437/aid/detectors/detect.py \
        --in_file "${IN_FILE}" \
        --out_file "${OUT_FILE}" \
        --task "${TASK}" \
        --lang "${LANG}" \
        --detectors "${DETECTORS[@]}"

      iter_end=$(date +%s)
      iter_runtime=$((iter_end - iter_start))

      iter_h=$((iter_runtime / 3600))
      iter_m=$(((iter_runtime % 3600) / 60))
      iter_s=$((iter_runtime % 60))

      echo "Iteration LANG=${LANG}, TASK=${TASK}, MODEL=${MODEL} finished in ${iter_h}h ${iter_m}m ${iter_s}s"
    done
  done
done

end_time=$(date +%s)
runtime=$((end_time - start_time))

hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

echo "Job finished at: $(date)"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"
