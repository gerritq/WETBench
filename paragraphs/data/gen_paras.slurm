#!/bin/bash
#SBATCH --job-name=ps-pt
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=nmes_cpu,cpu

source /scratch_tmp/users/k21157437/aid_env/bin/activate
which python 

LANG='pt'

python3 gen_paras.py ${LANG}
