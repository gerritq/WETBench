#!/bin/bash
#SBATCH --job-name=hp-vi
#SBATCH --output=../../sc_logs/vi/%j.out
#SBATCH --error=../../sc_logs/vi/%j.err
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# LAST RUN TOOK 65 mins (with one model)
# LANGUAGE="vi"
# MODELS=("vinai/phobert-base") # "prajjwal1/bert-small" "google-bert/bert-base-multilingual-cased" "FacebookAI/xlm-roberta-base"
# # MODELS=("prajjwal1/bert-small") # "prajjwal1/bert-small" "google-bert/bert-base-multilingual-cased"
# DATASUBSETS=("default" "mpairs")
# LEARNING_RATES=(5e-5 1e-5 1e-6)
# BATCH_SIZES=(16 32)
# EPOCHS=(3 5)
# WEIGHT_DECAY=(0 0.001)
# SEEDS=(42 2025)

source /scratch_tmp/users/k21157437/aid_env/bin/activate
which python 

nvidia-smi

# one model with the below configs takes 1 hour
LANGUAGE="vi"
MODELS=("FacebookAI/xlm-roberta-base" "google-bert/bert-base-multilingual-cased") # "FacebookAI/xlm-roberta-base" 
DATASUBSETS=("mpairs")
LEARNING_RATES=(5e-5 1e-5 5e-6 1e-6)
BATCH_SIZES=(16 32)
EPOCHS=(2 4 6)
WEIGHT_DECAY=(0 0.01)
SEEDS=(42 2025)
WARMUP_RATIO=.1

start_time=$(date +%s)

for MODEL in "${MODELS[@]}"; do
  for DATASUBSET in "${DATASUBSETS[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
      for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for EPOCH in "${EPOCHS[@]}"; do
          for WD in "${WEIGHT_DECAY[@]}"; do
            for SEED in "${SEEDS[@]}"; do
              MODEL_NAME=$(echo $MODEL | sed 's/\//_/g')  # Replace slashes for filenames
              LOG_FILE="../../sc_logs/${LANGUAGE}/${MODEL_NAME}_${DATASUBSET}_${LR}_${BATCH_SIZE}_${EPOCH}_${WD}_${SEED}.log"

              echo "Running experiment with lang=$LANGUAGE model=$MODEL, dataset=$DATASUBSET, lr=$LR, batch_size=$BATCH_SIZE, epochs=$EPOCH, weight_decay=$WD, seed=$SEED"

              run_start_time=$(date +%s)
              python3 train.py \
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

              if [[ $? -ne 0 ]]; then
                echo "Erros in training ..."
                exit 1
              fi

              echo "Experiment completed: model=$MODEL, dataset=$DATASUBSET, lr=$LR, batch_size=$BATCH_SIZE, epochs=$EPOCH, weight_decay=$WD, seed=$SEED"
              echo "Run time: $(($elapsed_time / 60)) minutes and $(($elapsed_time % 60)) seconds"
              #exit 0
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
