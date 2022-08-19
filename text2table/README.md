Main Project Script
========================================

This is the main script for the project. The folder contains the whole details about the task. `data` contains
the code that loads the data from the created final dataset and adds the special tokens in between different categories
of clinical notes. `models` contains codes for the final model architecture, the pre-tokenizer of the model, and the 
model training & evaluation. `metrics` contains the evaluation metrics for the model. The rest of the files, `config.yaml`
contains all parameters for model's training & evaluation. Any changes on the experiment should be made inside this file.
`ds_config.json` and `hostfile` are scripts for activating the deepspeed for training. Use it properly when neccessary (based on
number of GPUs & memory of each GPU). Futher details will be explained in each folder.
