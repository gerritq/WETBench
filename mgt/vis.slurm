#!/bin/bash
#SBATCH --job-name=vis
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --gres=gpu:1

export HF_HOME="/scratch/prj/inf_nlg_ai_detection/scratch_tmp/.cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

source /scratch/users/k21157437/aid_venv/bin/activate
which python

LANG='en'

python vis.py ${LANG}
