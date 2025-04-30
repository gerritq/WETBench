#!/bin/bash
#SBATCH --job-name=cat-hl-vi
#SBATCH --output=../../logs/%A.out
#SBATCH --error=../../logs/%A.err
#SBATCH --time=00:30:00


LANG="vi"

cat ../../data/${LANG}/processing/x_${LANG}_html_?.jsonl > ../../data/${LANG}/processing/2_${LANG}_html.jsonl

wc -l ../../data/${LANG}/processing/2_${LANG}_html.jsonl
