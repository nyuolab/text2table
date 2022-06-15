# Reference;
# HF Fine-tune Longformer Encoder-Decoder [tutorial](https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing#scrollTo=o9IkphgF-90-)
import datasets
from transformers import (LEDTokenizerFast, LEDForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer)
import os, shutil, logging


# Specify the directory where the pretokenized data are stored: train & validation sets
ckpt_dir_train = '/gpfs/data/oermannlab/users/cz1906/text2table/text2table/data/pretokenized/train/'
ckpt_dir_val = '/gpfs/data/oermannlab/users/cz1906/text2table/text2table/data/pretokenized/val/'

# If the pretokenized data are exists, load it directly from the disk (time-saving)
# If not, tokenized the text for model and store it for faster reuse
if os.path.exists(ckpt_dir_train) and os.path.exists(ckpt_dir_val):

    # Load the pre-tokenzied training dataset
    train_dataset = datasets.load_from_disk(ckpt_dir_train)
    # Load the pre-tokenized validation dataset
    val_dataset = datasets.load_from_disk(ckpt_dir_val)

else:

    # Load preprocessed data: with special tokens added
    train_dataset = datasets.load_dataset('../data/dataset_loading_script.py', split='train')
    val_dataset = datasets.load_dataset('../data/dataset_loading_script.py', split='validation')

    # Load tokenizer for the LED model
    tokenizer = LEDTokenizerFast.from_pretrained("allenai/led-base-16384")
    # Add special tokens to the LED model
    # As we want to represent the table as a sequence: separation tokens are added
    tokenizer.add_special_tokens({"additional_special_tokens": ["<COL>", "<ROW>"]})


    # Define the maximum input and output sequence length
    max_input_length = 8192
    max_output_length = 512 
    # Define the processing function so that the data will match the correct model format
    def process_data_to_model_inputs(batch):
        # Tokenize the input text
        inputs = tokenizer(
            batch['TEXT'],
            padding='max_length',
            truncation=True,
            max_length=max_input_length,
        )
        # Tokenize the output text
        outputs = tokenizer(
            batch['TABLE'],
            padding='max_length',
            truncation=True,
            max_length=max_output_length,
        )
        # Tokenize the head column for decoder
        head = tokenizer(
            batch['HEADER'],
        )

        # Assign the input IDs and corresponding attention mask
        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        # Assign the header IDs
        batch["decoder_input_ids"] = head.input_ids

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


    # Define the batch size & Num of CPUs
    batch_size = 32
    num_cpu = 32
    # Preprocess(Tokenize) the input data: Training set
    train_dataset = train_dataset.map(
        function=process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        num_proc=num_cpu,
        remove_columns=["DOB", "SEX", "ADMITTIME"],
    )
    # Preprocess(Tokenize) the input data: Validation set
    val_dataset = val_dataset.map(
        function=process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        num_proc=num_cpu,
        remove_columns=["DOB", "SEX", "ADMITTIME"],
    )

    # Convert the dataset to Pytorch format for LED
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )
    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )

    # Creat a directory to store the pretokenized data for training set
    os.makedirs(ckpt_dir_train)
    # Save
    train_dataset.save_to_disk(ckpt_dir_train)
    logging.info(f'saved tokenized training dataset to {ckpt_dir_train}')

    # Creat a directory to store the pretokenized data for validation set
    os.makedirs(ckpt_dir_val)
    # Save
    val_dataset.save_to_disk(ckpt_dir_val)
    logging.info(f'saved tokenized validation dataset to {ckpt_dir_val}')



# Initialize the model
model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384", gradient_checkpointing=True)
# Add special tokens to the LED model decoder
model.resize_token_embeddings(len(tokenizer))
# Setup the model's hyperparameters
model.config.num_beams = 2
model.config.max_length = 512
model.config.min_length = 0
model.config.length_penalty = 1.0
model.config.early_stopping = True


# Modify training parameters
training_args = Seq2SeqTrainingArguments(
    output_dir="../../models/",
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    fp16=True,
    logging_steps=5,
    eval_steps=600,
    save_steps=600,
    save_total_limit=2,
    gradient_accumulation_steps=4,
)


# Load the HuggingFace pre-defined "rouge" metric for evaluation
rouge = datasets.load_metric("rouge")

# Define the metric function for evalutation
def compute_metrics(pred):
    # Prediction IDs
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    # Prepare the data for evaluation (as Text2Table task, we care about the special tokens)
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=False)
    # Compute the rouge evaluation results
    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid
    # Return the results
    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start the training
trainer.train()
