#!/bin/bash

#SBATCH --job-name=timellama
#SBATCH --output=outputs/output-run-nc.txt
#SBATCH --mail-user=koerber@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --time=20:00:00
#SBATCH --mem=45000
#SBATCH --gres=gpu:1
#SBATCH --partition=students

srun python3 src/run.py
