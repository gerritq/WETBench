#!/bin/bash
#SBATCH --job-name=query-pt
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err
#SBATCH --partition=nmes_cpu,cpu
#SBATCH --time=11:00:00
#SBATCH --mem=10GB

source /scratch_tmp/users/k21157437/aid_env/bin/activate
which python 


LANG="vi"
IN_FILE="../data/${LANG}/1_${LANG}_latest_articles.jsonl"
OUT_FILE="../data/${LANG}/2_${LANG}_html.jsonl"
TOTAL_COUNT=100000

python 2_query2.py --lang $LANG \
    --in_file $IN_FILE \
    --out_file $OUT_FILE \
    --total_count $TOTAL_COUNT

