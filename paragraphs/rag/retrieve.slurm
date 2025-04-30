#!/bin/bash
#SBATCH --job-name=retrieve
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=nmes_gpu,gpu 
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB

start_time=$(date +%s)
echo "Job started at: $(date)"

nvidia-smi

source /scratch/users/k21157437/aid_venv/bin/activate
which python

LANGS=("pt")
SUBSETS=("extend")

python retrieve2.py --subsets "${SUBSETS[@]}" --langs "${LANGS[@]}"

end_time=$(date +%s)
runtime=$((end_time - start_time))

hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

echo "Job finished at: $(date)"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"