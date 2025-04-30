#!/bin/bash
#SBATCH --job-name=fewsh-en
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=nmes_cpu,cpu

LANG="en"
DS="paras"

python3 get_few_shots.py $LANG $DS > ../prompts/${LANG}/few_shots_paras.txt
