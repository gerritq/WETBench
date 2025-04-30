#!/bin/bash
#SBATCH --job-name=nrevs-vi
#SBATCH --output=/dev/null
#SBATCH --error=../../logs/%j.err
#SBATCH --time=3:00:00
#SBATCH --partition=nmes_cpu,cpu

source /scratch_tmp/users/k21157437/aid_env/bin/activate

LANG='vi'

python3 1_get_nrevs.py $LANG

