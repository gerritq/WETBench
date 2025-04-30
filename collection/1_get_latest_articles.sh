#!/bin/bash
#SBATCH --job-name=late-pt
#SBATCH --output=/dev/null
#SBATCH --error=../../logs/%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=cpu

LANG='en'

python3 1_get_latest_articles.py $LANG

# chek number of rows
wc -l ../../data/${LANG}/processing/1_${LANG}_latest_articles.jsonl
