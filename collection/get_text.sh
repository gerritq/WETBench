#!/bin/bash
#SBATCH --job-name=text-vi
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=20GB
#SBATCH --partition=nmes_cpu,cpu

source /scratch_tmp/users/k21157437/aid_env/bin/activate
which python 

LANG='vi'

python3 get_text.py ${LANG}