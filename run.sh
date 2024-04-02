#!/bin/bash

#SBATCH --job-name=timellama
#SBATCH --output=outputs/output-run1.txt
#SBATCH --mail-user=koerber@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH --mem=40000
#SBATCH --gres=gpu:1
#SBATCH --partition=students

srun python3 src/run.py
