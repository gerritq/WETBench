#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --output=../../logs/%A_%a.out
#SBATCH --error=../../logs/%A_%a.err
#SBATCH --partition=nmes_cpu,cpu
#SBATCH --time=12:00:00
#SBATCH --mem=15G
#SBATCH --array=0-5

source /scratch_tmp/users/k21157437/aid_env/bin/activate

python compare.py
