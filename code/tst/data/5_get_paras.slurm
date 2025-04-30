#!/bin/bash
#SBATCH --job-name=paras-en
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=nmes_cpu,cpu


source /scratch_tmp/users/k21157437/aid_env/bin/activate

LANG="en"
DS="mpairs"

python3 5_get_paras.py $LANG $DS
