#!/bin/bash

#SBATCH --job-name=timellama
#SBATCH --output=outputs/output-run0.txt
#SBATCH --mail-user=koerber@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --time=00:50:00
#SBATCH --mem=40000
#SBATCH --gres=gpu:1
#SBATCH --partition=students

srun python3 src/run.py
