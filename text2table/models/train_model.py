import datasets
from transformers import LEDTokenizerFast, LEDForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer


# Load preprocessed data
train_dataset = datasets.load_dataset('./dataset_loading_script.py', split='train')
val_dataset = datasets.load_dataset('./dataset_loading_script.py', split='validation')

# Load tokenizer for the LED model
tokenizer = LEDTokenizerFast.from_pretrained("allenai/led-base-16384")
# Add special tokens to the LED model
# As we want to represent the table as a sequence: separation tokens are added
tokenizer.add_special_tokens({"additional_special_tokens": ["<COL>", "<ROW>"]})


# Define the maximum input and output sequence length
max_input_length = 8000
max_output_length = 512

# Define the input processing function so that the data will match the correct model format
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
    # Assign the input 
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask








# Initialize the model
model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")


# Modify training parameters
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/",
    do_train=True,
    do_eval=True,
    per_gpu_train_batch_size=8,
    per_gpu_eval_batch_size=64,
    num_train_epochs=0.5, 
    logging_steps=500,
    logging_first_step=True,
    save_steps=1000,
    evaluation_strategy = "epoch", 
    learning_rate=2e-5,
    weight_decay=0.01,
)

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Start the training
trainer.train()
