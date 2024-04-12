#!/bin/bash

#SBATCH --job-name=timellama
#SBATCH --output=outputs/output-run-nc.txt
#SBATCH --mail-user=[insert-email-here]
#SBATCH --mail-type=ALL
#SBATCH --time=20:00:00
#SBATCH --mem=45000
#SBATCH --gres=gpu:1
#SBATCH --partition=students

srun python3 src/run.py
