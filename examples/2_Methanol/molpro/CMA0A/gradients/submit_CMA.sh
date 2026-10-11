#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=benchmarker
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=10G

source /home/mel64643/.bashrc
source activate CMA
python -u main.py > CMA.out



