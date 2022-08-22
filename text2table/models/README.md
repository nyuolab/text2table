Final Model and Experiment
==============================

The final model architecture and experiment for the text2table task. The model utlized is a
hierarchical-structured efficient transformer, Longformer-Encoder-Decoder (LED), and the
whole implementation is based on the pre-trianed LED (LED-Base) available on HuggingFace.
The pre-trained tokenizer and model are loaded directly from the HuggingFace and used for
fine-tune for the task.


Requirements
------------------
Make sure to run `make requirements` before running the actual model. And you can also run
`make test_environment` to see you have met the requirments.


Pre-Tokenize for Model
-------------------------
All input and output data are tokenized for model use. And the implementation is available in the file
`tokenizer.py`. To tokenize the data, one can simply run the `train_model.py` file, which also
includes the implemenation of the experiment, LED. All the tokenized data will be saved in
the parent directory, inside the `text2table/data/pretokenized/`, for time-saving whenever reuses are needed.
There are two modes available, one is the protokenization for MVP model, called `minimum`. Another is the
protokenization for final model, called `full`.


Model Archiecture
-------------------------
The final model architecture for the task, a hierarchical-structured efficient transformer, Longformer-Encoder-Decoder (LED),
is implemented in `modeling_hierarchical_led.py` file. The file implements the LEDForConditionalGeneration from Huggingface LED,
with the hierarchical-structure being added and modify. Moreover, in order to accomendate this structure, the file `data_collator.py`
is used to rewrite the whole data collator for the updated model's structure for training and evluating.


Experiment
--------------------
The experiment set-up (include training & evaluation activation) can be found at the `train_model.py`file. The process is implemented 
based on the `Trainer` and `TrainerArguments` modules available on the HuggingFace. This file is the main file for this whole project,
and it manipulates the process from dataset loading, dataset processed, dataset pretokenzed, model activation, and model evaluation.
Once all the requirments is satisfied, one can start fine-tuning for the text2table task by runining `python3 train_model.py`. Checkpoints,
and trained models can be found in the folder `../../models` once experiments are finished. Please make sure the "wandb" is activated for 
experiment results. Also, as mentioned in the main folder, please accommodate any parameter changes in the `../config.yaml` file.


--------

The model is completed and tested on multi-CPUs/GPUs set-up. Please arrange accordingly with the deepspeed implementation available in the parant folder.
