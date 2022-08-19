Data Loading & Processing
========================================

`dataset_loading_script.py` is the file that used to load and process the final dataset.

There are three modes for dataset loading. `minimum` mode will load the dataset for the MVP model.
`full` mode will load the final dataset for the final model. `dev` will load the partial final
dataset for the final model (as it is a development dataset, it is relatively small and representative,
helping faster and easier experiment testing & debugs)

To be able to read and differentiate different categories of clinical notes and mutiple notes under one category, 
the file also processes the dataset by adding model recognized special tokens in between texts. The processed data
will later be passed into the model's tokenizer for training.
