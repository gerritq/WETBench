#!/bin/bash
#SBATCH --job-name=en-acc
#SBATCH --output=../../sc_logs/pt/%j.out
#SBATCH --error=../../sc_logs/pt/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpu


python3 test_acc.py