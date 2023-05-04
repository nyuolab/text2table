# Dataset

The code files for creating the dataset for experiments. The dataset we use is
MIMIC-III (Medical Information Mart for Intensive Care III), and more details can be acccessed at
[MIMIC-III](https://mimic.mit.edu/docs/iii/). As the original MIMIC-III dataset
contains 26 tables and approximately hundreds of columns, we need to obtain and organize only data we use. 
Some samples clinicial notes for patients can be accessed at `sample_patient_data/`. 
Please note that the samples in the "after_preprocessing" and the samples in the "before_preprocessing" are different. 

## Requirements

Make sure to run `make requirements` before running any code. And you can also run
`make test_environment` to see you have met the requirments.

## Create the dataset

To create the dataset, please make sure to download the original MIMIC-III dataset and change the path in `dataset_v2.py`.
Then, you can use `python3 dataset_v2.py` to create the dataset.
