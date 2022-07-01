# Reference;
# HF Fine-tune Longformer Encoder-Decoder [tutorial](https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing#scrollTo=o9IkphgF-90-)
import datasets
from transformers import (LEDTokenizerFast, LEDForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, LEDConfig)
import os, shutil, logging, wandb
from tokenizer import tokenize
from omegaconf import OmegaConf

# Initialize wandb
wandb.init(project="text2table")

# Load the configuration
conf = OmegaConf.load("../config.yaml")

# Specify the directory where the pretokenized data are stored: train & validation sets
ptk_dir_train = conf.tokenizer.ptk_dir_train
ptk_dir_val = conf.tokenizer.ptk_dir_val


# Load tokenizer for the LED model
tokenizer = LEDTokenizerFast.from_pretrained("allenai/led-base-16384")
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

# Convert the dataset to Pytorch format for LED
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)


# Setup the model arguments
model_args = LEDConfig(
    num_beams=conf.model.num_beams,
    max_length=conf.model.max_length,
    min_length=conf.model.min_length,
    length_penalty=conf.model.length_penalty,
    early_stopping=conf.model.early_stopping,
)
# Initialize the model
model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384", config=model_args)
# Add special tokens to the LED model
model.resize_token_embeddings(len(tokenizer))


# Declare the training arguments
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
    run_name=conf.trainer.run_name,
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