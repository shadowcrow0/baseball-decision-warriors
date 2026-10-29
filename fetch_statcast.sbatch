#!/bin/bash
#SBATCH -p compute1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 06:00:00
#SBATCH -J statcast
#SBATCH -o statcast_%j.log

module load anaconda3/2024.10-1
source ~/pyb_env/bin/activate
cd ~/baseball
python src/fetch_statcast.py 2025-03-27 2025-09-28 data/statcast_2025
