#!/bin/bash
#SBATCH --job-name=cat-en
#SBATCH --output=../../logs/%A.out
#SBATCH --error=../../logs/%A.err
#SBATCH --time=00:30:00
#SBATCH --partition=cpu

source /scratch_tmp/users/k21157437/aid_env/bin/activate

LANG="en"

cat ../../data/${LANG}/processing/temp/*_${LANG}_crawl.jsonl > ../../data/${LANG}/processing/2_${LANG}_crawl.jsonl

# rm ../data/${LANG}/processing/temp/j?_${LANG}_crawl.jsonl
#wc -l ../../data/${LANG}/processing/2_${LANG}_crawl.jsonl

# wc -l ../../data/${LANG}/processing/1_${LANG}_nrevs.jsonl

# wc -l ../../../neutral/out/${LANG}/03_${LANG}nrevscrawl.jsonl
