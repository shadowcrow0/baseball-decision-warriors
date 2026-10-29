#!/bin/bash
#SBATCH -p compute1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 08:00:00
#SBATCH -J pyb_fetch
#SBATCH -o pyb_fetch_%j.log

module load anaconda3/2024.10-1
source ~/pyb_env/bin/activate
cd ~/baseball
set -e
python src/check_sources.py
python src/fetch_draft.py 2005 2025 data/draft.parquet
python src/fetch_war.py 2005 2025 data/
python src/fetch_statcast.py 2015-04-01 2015-10-04 data/statcast_2015
python src/build_features.py data/
