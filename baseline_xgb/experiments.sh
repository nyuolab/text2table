#!/bin/bash
#SBATCH --job-name=text2table_xgb
#SBATCH --output=/gpfs/data/oermannlab/users/yangz09/summer/text2table/log/xgb.out
#SBATCH --error=/gpfs/data/oermannlab/users/yangz09/summer/text2table/log/xgb.err
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ZihaoGavin.Yang@nyulangone.org

source ~/.bashrc
pwd
conda activate text2table

python3 new_xgb.py --all=y --tokenizer=bag_of_words