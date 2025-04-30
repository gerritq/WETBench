#!/bin/bash
#SBATCH --job-name=ft-en
#SBATCH --output=../../sc_logs/en/%j.out
#SBATCH --error=../../sc_logs/en/%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:30:00


nvidia-smi
# Define variables
LANGUAGE="en"
DSUBSET="ft"
MODEL="FacebookAI/roberta-base"
EPOCHS=15
BATCH_SIZE=32
LEARNING_RATE=1e-06
SEED=2025
WEIGHT_DECAY=0.01

echo "Running fine-tuning for language=$LANGUAGE on dataset=$DSUBSET with model=$MODEL..."
start_time=$(date +%s)

python3 train.py \
  --language $LANGUAGE \
  --dsubset $DSUBSET \
  --model $MODEL \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --seed $SEED \
  --weight_decay $WEIGHT_DECAY  2>&1

# Calculate elapsed time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Fine-tuning completed in $(($elapsed_time / 60)) minutes and $(($elapsed_time % 60)) seconds."
