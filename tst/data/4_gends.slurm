#!/bin/bash
#SBATCH --job-name=gends-vi-mp
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=nmes_cpu,cpu

source /scratch_tmp/users/k21157437/aid_env/bin/activate

LANG="vi"
DS="mpairs"

# LANG="vi"
# DS="mpairs"

python3 4_gends.py $LANG $DS
