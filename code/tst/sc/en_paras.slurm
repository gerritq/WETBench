#!/bin/bash
#SBATCH --job-name=hp-enp
#SBATCH --output=../../sc_logs/en/%j.out
#SBATCH --error=../../sc_logs/en/%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=rental_gpu
# SBATCH --partition=nmes_gpu,gpu
#SBATCH --gres=gpu:1

export HF_HOME="/scratch/prj/inf_nlg_ai_detection/scratch_tmp/.cache"

source /scratch/users/k21157437/aid_venv/bin/activate
which python

nvidia-smi

LANGUAGE="en"
MODELS=("FacebookAI/roberta-base")
DATASUBSETS=("paras")
LEARNING_RATES=(5e-05 1e-05 5e-06 1e-06)
BATCH_SIZES=(16 32)
EPOCHS=(3 5 7)
WEIGHT_DECAY=(0 0.01)
SEEDS=(2025)
WARMUP_RATIO=.1

start_time=$(date +%s)

for MODEL in "${MODELS[@]}"; do
  for DATASUBSET in "${DATASUBSETS[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
      for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for EPOCH in "${EPOCHS[@]}"; do
          for WD in "${WEIGHT_DECAY[@]}"; do
            for SEED in "${SEEDS[@]}"; do
              MODEL_NAME=$(echo $MODEL | sed 's/\//_/g')
              LOG_FILE="../../sc_logs/${LANGUAGE}/${MODEL_NAME}_${DATASUBSET}_${LR}_${BATCH_SIZE}_${EPOCH}_${WD}_${SEED}.log"

              echo "Running experiment with lang=$LANGUAGE model=$MODEL, dataset=$DATASUBSET, lr=$LR, batch_size=$BATCH_SIZE, epochs=$EPOCH, weight_decay=$WD, seed=$SEED"

              run_start_time=$(date +%s)
              python train.py \
                --language $LANGUAGE \
                --dsubset $DATASUBSET \
                --model $MODEL \
                --epochs $EPOCH \
                --batch_size $BATCH_SIZE \
                --learning_rate $LR \
                --weight_decay $WD \
                --warmup_ratio $WARMUP_RATIO \
                --seed $SEED  2>&1 # > $LOG_FILE
              run_end_time=$(date +%s)
              elapsed_time=$((run_end_time - run_start_time))

              echo "Experiment completed: model=$MODEL, dataset=$DATASUBSET, lr=$LR, batch_size=$BATCH_SIZE, epochs=$EPOCH, weight_decay=$WD, seed=$SEED"
              echo "Run time: $(($elapsed_time / 60)) minutes and $(($elapsed_time % 60)) seconds"
              # exit 0
            done
          done
        done
      done
    done
  done
done

end_time=$(date +%s)
total_elapsed_time=$((end_time - start_time))
echo "Total runtime: $(($total_elapsed_time / 60)) minutes and $(($total_elapsed_time % 60)) seconds"
