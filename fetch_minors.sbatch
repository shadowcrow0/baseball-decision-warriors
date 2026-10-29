#!/bin/bash
#SBATCH -p compute1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH -t 03:00:00
#SBATCH -J minors
#SBATCH -o minors_%j.log

module load anaconda3/2024.10-1
source ~/pyb_env/bin/activate
cd ~/baseball
set -e
python src/fetch_minors.py 2006 2026 data/
python src/profile_variables.py data/ reports/
