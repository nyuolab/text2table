# Reference;
# HF Fine-tune Longformer Encoder-Decoder [tutorial](https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing#scrollTo=o9IkphgF-90-)
import datasets
import torch
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, LEDConfig)
from transformers.utils.logging import set_verbosity_debug
import os, socket, wandb
from tokenizer import tokenize
from datasets import load_metric
from omegaconf import OmegaConf
from modeling_hierarchical_led import HierarchicalLEDForConditionalGeneration
from data_collator import data_collator


# Function to count the number of parameters in the encoder of the model
# Since the dataset is large, we will freeze the encoder weights to allow for fine-tuning
def count_param(m):
    pre_sum=0
    for param in m.parameters():
        if param.requires_grad == True:
            pre_sum+=param.numel()
    return pre_sum

##########################START##########################
# Load the configuration
conf = OmegaConf.load("../config.yaml")

# Initialize wandb
wandb.init(project="text2table", group=conf.trainer.group, 
name=conf.trainer.run_name + str(socket.gethostname()) + "_" + os.environ["LOCAL_RANK"], mode=conf.trainer.wandb_mode)

# Set the verbosity lebel for the huggingface transformers's root logger
if conf.trainer.debug:
    set_verbosity_debug()

# Specify the directory where the pretokenized data are stored: train & validation sets
ptk_dir_train = conf.tokenizer.ptk_dir_train
ptk_dir_val = conf.tokenizer.ptk_dir_val


# training script for the minimum dataset
if conf.dataset.version == "minimum":
    # Load tokenizer for the LED model
    tokenizer = AutoTokenizer.from_pretrained('patrickvonplaten/led-large-16384-pubmed')
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
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )
    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )


    # Initialize the model
    config = LEDConfig.from_pretrained('patrickvonplaten/led-large-16384-pubmed')
    model = AutoModelForSeq2SeqLM.from_pretrained('patrickvonplaten/led-large-16384-pubmed', config=config)
    
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
        include_inputs_for_metrics=True
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


# training script for the full dataset
elif conf.dataset.version == "full" or conf.dataset.version == "dev":
    # Load tokenizer for the LED model
    tokenizer = AutoTokenizer.from_pretrained('allenai/led-base-16384')
    # Add special tokens to the LED model
    # As we want to represent the table as a sequence: separation tokens are added
    tokenizer.add_special_tokens({"additional_special_tokens": ["<CEL>", "<NTE>", 
    "<NUR>", "<DIS>", "<ECH>", "<ECG>", "<RAD>", "<PHY>", "<GEN>", "<RES>", "<NUT>", 
    "<GENDER>", "<DOB>", "<CPT_CD>", "<DRG_CODE>", "<DIAG_ICD9>", "<LAB_MEASUREMENT>",
    "<PRESCRIPTION>", "<PROC_ICD9>"]})

    # If the pretokenized data are exists, load it directly from the disk (time-saving)
    # If not, tokenized the text for model and store it for faster reuse (Call Tokenizer in the same directory)
    if not (os.path.exists(ptk_dir_train) and os.path.exists(ptk_dir_val)):
        # Pre-tokenize the input text & save the result in the directory
        tokenize()

    # Load the pre-tokenzied training dataset
    train_dataset = datasets.load_from_disk(ptk_dir_train)
    # Load the pre-tokenized validation dataset
    val_dataset = datasets.load_from_disk(ptk_dir_val)

    # Convert and save the dataset to the torch format for the model
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "decoder_input_ids", "global_attention_mask", "labels"],
    )
    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "decoder_input_ids", "global_attention_mask", "labels"],
    )

    # Initialize the model
    model = HierarchicalLEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
    
    # Add special tokens to the LED model
    model.resize_token_embeddings(len(tokenizer))

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
        eval_accumulation_steps=conf.trainer.eval_accumulation_steps,
        save_steps=conf.trainer.save_steps,
        save_total_limit=conf.trainer.save_total_limit,
        gradient_accumulation_steps=conf.trainer.gradient_accumulation_steps,
        include_inputs_for_metrics=True,
        deepspeed=conf.trainer.deepspeed,
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
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Freeze the model's encoder weights
    # pre-freeze
    print("pre_freeze param: ",count_param(model))
    # freeze
    for param in model.led.encoder.parameters():
        param.requires_grad = False
    for param in model.led.encoder.embed_tokens.parameters():
        param.requires_grad = True
    # post-freeze
    print("post_freeze param: ",count_param(model))

    # Start the training
    trainer.train()
