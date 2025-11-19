#!/bin/bash

#SBATCH --job-name=Bootstrap
#SBATCH --array=0-49
#SBATCH --time=28:00:00
#SBATCH -p batch
#SBATCH -n 1
#SBATCH --mem=25G
#SBATCH --cpus-per-task=4
#SBATCH --output=output_files/job_output_%a.txt
#SBATCH --error=error_files/job_error_%a.txt

module load anaconda/2021.11
source activate my_env

mkdir -p output_files
mkdir -p error_files
python Running_through_bootstrap_experiment.py