#!/bin/bash

#SBATCH --job-name=timellama
#SBATCH --output=outputs.txt
#SBATCH --mail-user=koerber@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --time=00:20:00
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --partition=students

srun python3 Xtemporal-reasoning/test_model.py