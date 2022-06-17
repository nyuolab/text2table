# Reference;
# HF Fine-tune Longformer Encoder-Decoder [tutorial](https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing#scrollTo=o9IkphgF-90-)
import datasets
from transformers import (LEDTokenizerFast, LEDForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer)
import os, shutil, logging
from tokenizer import tokenize


# Specify the directory where the pretokenized data are stored: train & validation sets
ckpt_dir_train = '../data/pretokenized/train/'
ckpt_dir_val = '../data/pretokenized/val/'


# Load tokenizer for the LED model
tokenizer = LEDTokenizerFast.from_pretrained("allenai/led-base-16384")
# Add special tokens to the LED model
# As we want to represent the table as a sequence: separation tokens are added
tokenizer.add_special_tokens({"additional_special_tokens": ["<COL>", "<ROW>", "<CEL>"]})


# If the pretokenized data are exists, load it directly from the disk (time-saving)
# If not, tokenized the text for model and store it for faster reuse (Call Tokenizer in the same directory)
if not (os.path.exists(ckpt_dir_train) and os.path.exists(ckpt_dir_val)):
    # Pre-tokenize the input text & save the result in the directory
    tokenize()


# Load the pre-tokenzied training dataset
train_dataset = datasets.load_from_disk(ckpt_dir_train)
# Load the pre-tokenized validation dataset
val_dataset = datasets.load_from_disk(ckpt_dir_val)

# Convert the dataset to Pytorch format for LED
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)


# Modify model & trainer parameters
gradient_checkpointing=True

predict_with_generate=True
evaluation_strategy="steps"
per_device_train_batch_size=2
per_device_eval_batch_size=2


# Initialize the model
model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384", gradient_checkpointing=gradient_checkpointing)
# Add special tokens to the LED model decoder
model.resize_token_embeddings(len(tokenizer))
# Setup the model's hyperparameters
model.config.num_beams = 2
model.config.max_length = 512
model.config.min_length = 0
model.config.length_penalty = 1.0
model.config.early_stopping = True


# Declare the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="../../models/",
    predict_with_generate=predict_with_generate,
    evaluation_strategy=evaluation_strategy,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    fp16=True,
    logging_steps=10,
    eval_steps=1000,
    save_steps=1000,
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