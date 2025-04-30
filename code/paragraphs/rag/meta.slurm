#!/bin/bash
#SBATCH --job-name=me-ptX
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=nmes_cpu,cpu 
#SBATCH --cpus-per-task=6
#SBATCH --mem=4GB

source /scratch/users/k21157437/aid_venv/bin/activate
which python

# rag-search-1
# API_KEY=""
# CX_ID=""

# rag-search-2
API_KEY=""
CX_ID=""

LANG='pt'
SUBSET='extend'
N=2700
# 2430 (en extend)
# 2700 pt first
#em
#N=2160 # 540

python meta2.py $LANG $SUBSET $N $API_KEY $CX_ID

# echo "Done with ${LANG} ${SUBSET}"
# sleep 180

# LANG='vi'
# SUBSET='first'
# N=270

# python meta2.py $LANG $SUBSET $N

# echo "Done with ${LANG} ${SUBSET}"