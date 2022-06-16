MVP Model
==============================

The minimum valuable product for the text2table task. The model utlized for this MVP is 
the Longformer-Encoder-Decoder (LED), and the whole implementation is based on the pre-
trianed LED available on HuggingFace. The pre-trained tokenizer and model are loaded
directly from the HuggingFace and used for fine-tune for our text2table task.

Requirements
------------------
Make sure to run `make requirements` before running the actual model. And you can also run
`make test_environment` to see you have met the requirments.

Datasets
-----------------
The datatset used for the MVP consists of the data concatenated from the tables `PATIENTS`,
`NOTEEVENTS`, and `D_ICD_DIAGNOSES`. And the concatenated is based on the Patient ID number
and Patient Specfici Admission ID number. The concatenated dataset consists of approximately
50000 rows of data with each row represents one patient for his/her specific admission. Since
this is a text2table task, the input data is the Discharge Summary of the clinical notes, and
the output columns are constraint for Patient Sex, Date of Birth, Admission Time, and ICD9 Code.

Pre-Tokenize for Model
-------------------------
All input and output data are tokenized for model use. And the implementation is available in the file
`tokenizer.py`. To tokenize the data, one can simply run the `train_model.py` file, which also
includes the implemenation and fine-tune of the MVP model, LED. All the tokenized data will be saved in
the parent directory, inside the `text2table/data/pretokenized/`, for time-saving whenever reuses are needed.

Fine-Tune
--------------------
The implementation of the model, LED, and the whole fine-tuning process can be found at the `train_model.py`
file. The fine-tuning is implemented based on the `Trainer` and `TrainerArguments` modules available on the 
HuggingFace. Once all the requirments is satisfied, one can start fine-tuning for the text2table task by runining 
`python3 train_model.py`.


--------

The MVP is completed and tested on multi-CPUs/GPUs set-up. Please arrange accordingly. For example, if one has 
4 GPUs for fine-tuning, one can activate the distributed training by using the 
`python3 -m torch.distributed.launch --nproc_per_node 4 train_model.py`。
