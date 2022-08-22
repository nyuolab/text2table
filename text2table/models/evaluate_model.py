from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
import datasets
import torch
from tokenizer import tokenize
from modeling_hierarchical_led import HierarchicalLEDForConditionalGeneration
from data_collator import data_collator
import os, socket, wandb
from omegaconf import OmegaConf
from datasets import load_metric

# Load the configuration
conf = OmegaConf.load("../config.yaml")

# Initialize wandb
wandb.init(project="text2table", group=conf.trainer.group, 
name=conf.trainer.run_name + str(socket.gethostname()), mode=conf.trainer.wandb_mode)
# + "_" + os.environ["LOCAL_RANK"]

# Specify the directory where the pretokenized data are stored
ptk_dir_val = conf.tokenizer.ptk_dir_val

if conf.dataset.version == "full" or conf.dataset.version == "dev":

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('../../models/trained_model/')

    # If the pretokenized data are exists, load it directly from the disk (time-saving)
    # If not, tokenized the text for model and store it for faster reuse (Call Tokenizer in the same directory)
    if not os.path.exists(ptk_dir_val):
        # Pre-tokenize the input text & save the result in the directory
        tokenize(train=False)

    # Load the pre-tokenized validation dataset
    val_dataset = datasets.load_from_disk(ptk_dir_val)

    # Convert and save the dataset to the torch format for the model
    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "decoder_input_ids", "global_attention_mask", "labels"],
    )

    # Load the model
    model = HierarchicalLEDForConditionalGeneration.from_pretrained('../../models/trained_model/')

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
        prediction_loss_only=conf.trainer.prediction_loss_only,
        evaluation_strategy=conf.trainer.evaluation_strategy,
        per_device_train_batch_size=conf.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=conf.trainer.per_device_eval_batch_size,
        logging_steps=conf.trainer.logging_steps,
        eval_steps=conf.trainer.eval_steps,
        save_steps=conf.trainer.save_steps,
        save_total_limit=conf.trainer.save_total_limit,
        gradient_accumulation_steps=conf.trainer.gradient_accumulation_steps,
        include_inputs_for_metrics=True,
    )

    #load custom metric
    main_metric = load_metric('../metrics/main_metric_script.py')
        
    def compute_metrics(EvalPrediction):
        predictions = EvalPrediction.predictions
        label_ids = EvalPrediction.label_ids
        inputs = EvalPrediction.inputs
        
        pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=False)
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=False)
        input_str=tokenizer.batch_decode(inputs[:, [1], 1], skip_special_tokens=False)
        print(input_str)

        # Compute the rouge evaluation results
        main_metric_output = main_metric.compute(predictions=pred_str,references=label_str,inputs=input_str)
        
        return main_metric_output


    # Initialize the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset
    )

    with torch.no_grad():
        # evaluate the model
        trainer.evaluate()
