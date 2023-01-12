#!/bin/bash
#SBATCH --job-name=text2table_xgb
#SBATCH --output=/gpfs/data/oermannlab/users/yangz09/summer/text2table/log/xgb.out
#SBATCH --error=/gpfs/data/oermannlab/users/yangz09/summer/text2table/log/xgb.err
#SBATCH --partition=cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ZihaoGavin.Yang@nyulangone.org

source ~/.bashrc
pwd
conda activate text2table

python3 new_xgb.py --all=n --tokenizer=bag_of_words --mode=train --task=icd9_proc
python3 new_xgb.py --all=n --tokenizer=bag_of_words --mode=predict_test --task=icd9_proc --partition=test
rm -rfv baseline_folder
python3 new_xgb.py --all=n --tokenizer=bag_of_words --mode=train --task=icd9_proc/cpt_cd
python3 new_xgb.py --all=n --tokenizer=bag_of_words --mode=predict_test --task=icd9_proc/cpt_cd --partition=test
rm -rfv baseline_folder
python3 new_xgb.py --all=n --tokenizer=bag_of_words --mode=train --task=icd9_proc/cpt_cd/drg_code
python3 new_xgb.py --all=n --tokenizer=bag_of_words --mode=predict_test --task=icd9_proc/cpt_cd/drg_code --partition=test
rm -rfv baseline_folder
python3 new_xgb.py --all=n --tokenizer=bag_of_words --mode=train --task=icd9_proc/cpt_cd/drg_code/gender
python3 new_xgb.py --all=n --tokenizer=bag_of_words --mode=predict_test --task=icd9_proc/cpt_cd/drg_code/gender --partition=test
rm -rfv baseline_folder
python3 new_xgb.py --all=n --tokenizer=bag_of_words --mode=train --task=icd9_proc/cpt_cd/drg_code/gender/expire_flag
python3 new_xgb.py --all=n --tokenizer=bag_of_words --mode=predict_test --task=icd9_proc/cpt_cd/drg_code/gender/expire_flag --partition=test
rm -rfv baseline_folder

python3 new_xgb.py --all=n --tokenizer=tfidf --mode=train --task=icd9_proc
python3 new_xgb.py --all=n --tokenizer=tfidf --mode=predict_test --task=icd9_proc --partition=test
rm -rfv baseline_folder
python3 new_xgb.py --all=n --tokenizer=tfidf --mode=train --task=icd9_proc/cpt_cd
python3 new_xgb.py --all=n --tokenizer=tfidf --mode=predict_test --task=icd9_proc/cpt_cd --partition=test
rm -rfv baseline_folder
python3 new_xgb.py --all=n --tokenizer=tfidf --mode=train --task=icd9_proc/cpt_cd/drg_code
python3 new_xgb.py --all=n --tokenizer=tfidf --mode=predict_test --task=icd9_proc/cpt_cd/drg_code --partition=test
rm -rfv baseline_folder
python3 new_xgb.py --all=n --tokenizer=tfidf --mode=train --task=icd9_proc/cpt_cd/drg_code/gender
python3 new_xgb.py --all=n --tokenizer=tfidf --mode=predict_test --task=icd9_proc/cpt_cd/drg_code/gender --partition=test
rm -rfv baseline_folder
python3 new_xgb.py --all=n --tokenizer=tfidf --mode=train --task=icd9_proc/cpt_cd/drg_code/gender/expire_flag
python3 new_xgb.py --all=n --tokenizer=tfidf --mode=predict_test --task=icd9_proc/cpt_cd/drg_code/gender/expire_flag --partition=test
rm -rfv baseline_folder
