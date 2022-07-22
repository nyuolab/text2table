# Reference;
# HF Fine-tune Longformer Encoder-Decoder [tutorial](https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing#scrollTo=o9IkphgF-90-)
import datasets
import torch
from transformers import (AutoTokenizer, LongT5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, LongT5Config)
import os, shutil, logging, wandb
from tokenizer import tokenize
from datasets import load_metric
import datetime
from text2table.logging_utils.logging_script import setup_logger
from omegaconf import OmegaConf


# Load the configuration
conf = OmegaConf.load("../config.yaml")
# Initialize wandb
wandb.init(project="text2table", group=conf.trainer.group, name=conf.trainer.run_name)


# Specify the directory where the pretokenized data are stored: train & validation sets
ptk_dir_train = conf.tokenizer.ptk_dir_train
ptk_dir_val = conf.tokenizer.ptk_dir_val

# Load tokenizer for the LED model
tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
# Add special tokens to the LED model
# As we want to represent the table as a sequence: separation tokens are added
tokenizer.add_special_tokens({"additional_special_tokens": ["<COL>", "<ROW>", "<CEL>"]})


# If the pretokenized data are exists, load it directly from the disk (time-saving)
# If not, tokenized the text for model and store it for faster reuse (Call Tokenizer in the same directory)
if not (os.path.exists(ptk_dir_train) and os.path.exists(ptk_dir_val)):
    # Pre-tokenize the input text & save the result in the directory
    tokenize()
    

# Load the pre-tokenzied training dataset
train_dataset = datasets.load_from_disk(ptk_dir_train)
# Load the pre-tokenized validation dataset
val_dataset = datasets.load_from_disk(ptk_dir_val)
# Prepare column header for the decoder
column_header = torch.LongTensor(train_dataset["decoder_input_ids"])

# Convert and save the dataset to the torch.Tensor format for the model
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)


# Initialize the model
model = LongT5Model.from_pretrained("google/long-t5-tglobal-base")
# Add special tokens to the LED model
model.resize_token_embeddings(len(tokenizer))

# Define whether we add column header to the decoder input
if (conf.trainer.use_decoder_header):
    model.prepare_decoder_input_ids_from_labels(column_header)

# modify model configuration
model.config.num_beams=conf.model.num_beams
model.config.max_length=conf.model.max_length
model.config.min_length=conf.model.min_length
model.config.length_penalty=conf.model.length_penalty
model.config.early_stopping=conf.model.early_stopping

# Declare the training pts
training_args = Seq2SeqTrainingArguments(
    gradient_checkpointing=conf.trainer.gradient_checkpointing,
    output_dir=conf.trainer.output_dir,
    predict_with_generate=conf.trainer.predict_with_generate,
    evaluation_strategy=conf.trainer.evaluation_strategy,
    per_device_train_batch_size=conf.trainer.per_device_train_batch_size,
    per_device_eval_batch_size=conf.trainer.per_device_eval_batch_size,
    fp16=conf.trainer.fp16,
    logging_steps=conf.trainer.logging_steps,
    eval_steps=conf.trainer.eval_steps,
    save_steps=conf.trainer.save_steps,
    save_total_limit=conf.trainer.save_total_limit,
    gradient_accumulation_steps=conf.trainer.gradient_accumulation_steps,
)

#load custom metric
cel_match = load_metric('../metrics/col_wise_metric_script.py')
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
    cel_match_output = cel_match.compute(predictions=pred_str,references=label_str,mode=[0,10,20])
    
    return cel_match_output

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Start the training
trainer.train()
