#!/bin/bash
#SBATCH --job-name=eval-all
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=10GB
#SBATCH --partition=nmes_cpu,cpu

source /scratch_tmp/users/k21157437/aid_env/bin/activate
which python 

LANG="en"
DS="default"
python gen_eval_set.py $LANG $DS

LANG="en"
DS="paras"
python gen_eval_set.py $LANG $DS

LANG="pt"
DS="default"
python gen_eval_set.py $LANG $DS

LANG="vi"
DS="mpairs"
python gen_eval_set.py $LANG $DS