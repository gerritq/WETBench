#!/bin/bash
#SBATCH --job-name=crawl-vi
#SBATCH --error=../../logs/%A-%a.err
#SBATCH --output=../../logs/%A-%a.out
#SBATCH --time=03:00:00
#SBATCH --partition=nmes_cpu
#SBATCH --array=0-1

source /scratch_tmp/users/k21157437/aid_env/bin/activate

python3 2_crawl.py vi

# cat ../../data/${LANG}/processing/temp/*_${LANG}_crawl.jsonl > ../../data/${LANG}/processing/2_${LANG}_crawl.jsonl

