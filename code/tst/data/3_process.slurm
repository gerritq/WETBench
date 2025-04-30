#!/bin/bash
#SBATCH --job-name=proc-vi
#SBATCH --output=../../logs/%A_%a.out
#SBATCH --error=../../logs/%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=nmes_cpu,cpu
#SBATCH --array=0-5

# SBATCH --time=10:00:00
# SBATCH --partition=nmes_cpu,cpu
# SBATCH --array=0-19

source /scratch_tmp/users/k21157437/aid_env/bin/activate

LANG="vi"

python3 3_process.py $LANG

# cat ../../data/${LANG}/processing/temp/j*_${LANG}_proc.jsonl > ../../data/${LANG}/processing/3_${LANG}_proc.jsonl

# if [[ $SLURM_ARRAY_TASK_COUNT -gt 1 ]]; then
#     cat ../data/${LANG}/processing/temp/j*_${LANG}_proc.jsonl > ../data/${LANG}/processing/3_${LANG}_proc.jsonl
# fi