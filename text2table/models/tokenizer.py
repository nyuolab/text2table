import datasets
from transformers import AutoTokenizer
import os, shutil, logging
from omegaconf import OmegaConf
import torch


# Define the tokenize function
def tokenize(train=True, val=True, test=True):

    # Load the configuration
    conf = OmegaConf.load("../config.yaml")

    # Specify the directory where the pretokenized data are stored: train & validation sets
    ptk_dir_train = conf.tokenizer.ptk_dir_train
    ptk_dir_val = conf.tokenizer.ptk_dir_val
    
    # Tokenize the mininum dataset
    if conf.dataset.version == "minimum":

        # Load preprocessed data: with special tokens added
        train_dataset = datasets.load_dataset('../data/dataset_loading_script.py', name='minimum', split='train')
        val_dataset = datasets.load_dataset('../data/dataset_loading_script.py', name='minimum', split='validation')

        # Load tokenizer for the LED model
        tokenizer = AutoTokenizer.from_pretrained('patrickvonplaten/led-large-16384-pubmed')
        # Add special tokens to the LED model
        # As we want to represent the table as a sequence: separation tokens are added
        tokenizer.add_special_tokens({"additional_special_tokens": ["<COL>", "<ROW>", "<CEL>"]})

        # Define the processing function so that the data will match the correct model format
        def process_data_to_model_inputs(batch):
            # Tokenize the input text
            inputs = tokenizer(
                batch['TEXT'],
                padding='max_length',
                truncation=True,
                max_length=conf.tokenizer.max_input_length,
            )
            # Tokenize the output text
            outputs = tokenizer(
                batch['TABLE'],
                padding='max_length',
                truncation=True,
                max_length=conf.tokenizer.max_output_length,
            )
            # Tokenize the head column for decoder
            head = tokenizer(
                batch['HEADER'],
                padding='max_length',
                max_length=conf.tokenizer.max_header_length,
            )

            # Assign the input IDs and corresponding attention mask
            batch["input_ids"] = inputs.input_ids
            batch["attention_mask"] = inputs.attention_mask
            # Assign the header IDs
            batch["decoder_input_ids"] = head.input_ids
            # Assign the attention mask for the header
            batch["decoder_attention_mask"] = head.attention_mask

            # create 0 global_attention_mask lists (0: token attends 'locally')
            batch["global_attention_mask"] = len(batch["input_ids"]) * [
                [0 for _ in range(len(batch["input_ids"][0]))]
            ]
            # Change the first token of all sequences to make it attend 'globally' (suggested by the original paper)
            batch["global_attention_mask"][0][0] = 1

            # Assign the output IDs
            batch["labels"] = outputs.input_ids

            # Ignore the PAD token
            batch["labels"] = [
                [-100 if token == tokenizer.pad_token_id else token for token in labels]
                for labels in batch["labels"]
            ]
            
            return batch


        # Preprocess(Tokenize) the input data: Training set
        train_dataset = train_dataset.map(
            function=process_data_to_model_inputs,
            batched=True,
            batch_size=conf.tokenizer.batch_size,
            num_proc=conf.tokenizer.num_cpu,
            remove_columns=["DOB", "SEX", "ADMITTIME", "ICD9"],
        )
        # Preprocess(Tokenize) the input data: Validation set
        val_dataset = val_dataset.map(
            function=process_data_to_model_inputs,
            batched=True,
            batch_size=conf.tokenizer.batch_size,
            num_proc=conf.tokenizer.num_cpu,
            remove_columns=["DOB", "SEX", "ADMITTIME", "ICD9"],
        )


    # Tokenize the full dataset
    elif conf.dataset.version == "full" or conf.dataset.version == "dev":
        
        # Load preprocessed data: with special tokens added
        if conf.dataset.version == "dev":
            if train:  
                train_dataset = datasets.load_dataset('../data/dataset_loading_script.py', name='dev', split='train')
            if val:
                val_dataset = datasets.load_dataset('../data/dataset_loading_script.py', name='dev', split='validation')
        else:
            if train:
                train_dataset = datasets.load_dataset('../data/dataset_loading_script.py', split='train')
            if val:
                val_dataset = datasets.load_dataset('../data/dataset_loading_script.py', split='validation')

        # Load tokenizer for the LED model
        tokenizer = AutoTokenizer.from_pretrained('allenai/led-base-16384')
        # Add special tokens to the LED model
        # As we want to represent the table as a sequence: separation tokens are added
        tokenizer.add_special_tokens({"additional_special_tokens": ["<CEL>", "<NTE>", 
        "<NUR>", "<DIS>", "<ECH>", "<ECG>", "<RAD>", "<PHY>", "<GEN>", "<RES>", "<NUT>", 
        "<GENDER>", "<DOB>", "<CPT_CD>", "<DRG_CODE>", "<DIAG_ICD9>", "<LAB_MEASUREMENT>"
        "<PRESCRIPTION>", "<PROC_ICD9>"]})

        # Define the processing function so that the data will match the correct model format
        def process_data_to_model_inputs(example):
            # Separate the input texts
            example["text_list"] = example["text"].split(" <text-sep> ")
            # Tokenize the input texts
            inputs = tokenizer(
                example['text_list'],
                padding='max_length',
                truncation=True,
                max_length=conf.tokenizer.max_input_length,
            )
            
            # Tokenize the output text
            output = tokenizer(
                example['label'],
                padding='max_length',
                truncation=True,
                max_length=conf.tokenizer.max_output_length,
            )
            
            # Tokenize the column header for decoder
            column_header = tokenizer(
                example['category'],
                add_special_tokens=False,
                padding='max_length',
                truncation=True,
                max_length=conf.tokenizer.max_header_length,
            )

            # Assign the input IDs and corresponding attention mask
            example["input_ids"] = inputs.input_ids
            example["attention_mask"] = inputs.attention_mask
            # Assign the header IDs
            decoder_input_ids = column_header.input_ids
            shifted_input_ids = [0] * len(decoder_input_ids)
            shifted_input_ids[1:] = decoder_input_ids[:-1].copy()
            shifted_input_ids[0] = 2
            example["decoder_input_ids"] = shifted_input_ids

            # create 0 global_attention_mask lists (0: token attends 'locally')
            example["global_attention_mask"] = len(example["input_ids"]) * [
                [0 for _ in range(len(example["input_ids"][0]))]
            ]

            # Change the first and the second token of all sequences to make it attend 'globally' (suggested by the original paper)
            example["global_attention_mask"][0][0] = 1

            # Assign the output IDs
            example["labels"] = output.input_ids

            # Ignore the PAD token
            example["labels"] = [-100 if token == tokenizer.pad_token_id else token for token in example["labels"]]
            
            return example

        if train:
            # Preprocess(Tokenize) the input data: Training set
            train_dataset = train_dataset.map(
                function=process_data_to_model_inputs,
                batched=False,
                num_proc=conf.tokenizer.num_cpu,
                remove_columns=["category", "label", "text"],
            ).remove_columns("text_list")

        if val:
            # Preprocess(Tokenize) the input data: Validation set
            val_dataset = val_dataset.map(
                function=process_data_to_model_inputs,
                batched=False,
                num_proc=conf.tokenizer.num_cpu,
                remove_columns=["category", "label", "text"],
            ).remove_columns("text_list")

    if train:
        # Creat a directory to store the pretokenized data for training set 
        os.makedirs(ptk_dir_train, exist_ok=True)
        # Save
        train_dataset.save_to_disk(ptk_dir_train)
        logging.info(f'saved tokenized training dataset to {ptk_dir_train}')

    if val:
        # Creat a directory to store the pretokenized data for validation set
        os.makedirs(ptk_dir_val, exist_ok=True)
        # Save
        val_dataset.save_to_disk(ptk_dir_val)
        logging.info(f'saved tokenized validation dataset to {ptk_dir_val}')
    