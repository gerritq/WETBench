#!/bin/bash
#SBATCH --job-name=gsums-all
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --partition=nmes_cpu,cpu
#SBATCH --time=03:30:00

source /scratch_tmp/users/k21157437/aid_env/bin/activate
which python 

LANGUAGES=("en" "pt" "vi")

for LANG in "${LANGUAGES[@]}"; do
    python gen_sums.py "$LANG"
done