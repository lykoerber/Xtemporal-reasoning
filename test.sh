#!/bin/bash

#SBATCH --job-name=timellama
#SBATCH --output=outputs/output-test.txt
#SBATCH --mail-user=koerber@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --time=00:10:00
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --partition=students

srun python3 test_model.py
