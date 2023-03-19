import argparse
import numpy as np
import pickle as pkl
from dataset_loading import loading_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    EvalPrediction, 
    set_seed
)
from models.bert import BertForSequenceClassification_corrloss
from models.roberta import RobertaForSequenceClassification_corrloss
from models.longformer import LongformerForSequenceClassification_corrloss
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, precision_recall_curve
from corr_metric import corr_score
import torch
import datasets
import wandb
import os

if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--top50', action='store_true')
    parser.add_argument('--tuning', action='store_true')
    parser.add_argument('--freeze', type=str, required=True)
    parser.add_argument('--corrloss', action='store_true')
    args = parser.parse_args()

    corrloss_postfix = "_corrloss" if args.corrloss else ""
    top50_postfix = "(top50)" if args.top50 else ""

    # set the model name and wandb group name and model path
    if args.model == "bert":
        model_name = 'emilyalsentzer/Bio_Discharge_Summary_BERT'
    elif args.model == "roberta":
        model_name = 'roberta-base'
    elif args.model == "longformer":
        model_name = 'allenai/longformer-base-4096'
    group = model_name.split("/")[-1] + "_" + args.freeze + corrloss_postfix
    model_path = group + "/"

    # initialize wandb
    if "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) == 0:
        wandb.init(project="text2data", entity="olab", group=group, name=args.task + top50_postfix)
    
    # set seed
    set_seed(42)

    # path
    data_dir="/gpfs/data/oermannlab/project_data/text2table/complete_v2/train_test_data/"

    # load the dataset
    [train, dev, test] = loading_dataset(data_dir, args.task.split('-'), args.top50)
    dataset = datasets.DatasetDict({"train":train,"test":test, "validation":dev})

    # get the labels
    labels = [label for label in dataset['train'].features.keys() if label not in ['TEXT', '__index_level_0__']]
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}
    
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # define the max length
    if model_name == "allenai/longformer-base-4096":
        max_length = 4096
    else:
        max_length = 512
    
    # preprocess for training
    def preprocess_data(examples):
        # take a batch of texts
        text = examples["TEXT"]
        # encode them
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(labels)))
        # fill numpy array
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()
        
        return encoding
    
    encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
    encoded_dataset.set_format("torch")

    # define the model
    if args.corrloss:
        if model_name == "emilyalsentzer/Bio_Discharge_Summary_BERT":
            model = BertForSequenceClassification_corrloss.from_pretrained(model_name, 
                                                            problem_type="multi_label_classification", 
                                                            num_labels=len(labels),
                                                            id2label=id2label,
                                                            label2id=label2id)
        elif model_name == "roberta-base":
            model = RobertaForSequenceClassification_corrloss.from_pretrained(model_name,
                                                            problem_type="multi_label_classification",
                                                            num_labels=len(labels),
                                                            id2label=id2label,
                                                            label2id=label2id)
        elif model_name == "allenai/longformer-base-4096":
            model = LongformerForSequenceClassification_corrloss.from_pretrained(model_name,
                                                            problem_type="multi_label_classification",
                                                            num_labels=len(labels),
                                                            id2label=id2label,
                                                            label2id=label2id)
        else:
            # raise an error
            raise ValueError("Model not supported!")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                            problem_type="multi_label_classification", 
                                                            num_labels=len(labels),
                                                            id2label=id2label,
                                                            label2id=label2id)
    
    # set the training argument
    batch_size = 32
    metric_name = "macro_f1"
    output_dir = model_path + args.task + "-output" + top50_postfix
    training_args = TrainingArguments(
        output_dir,
        logging_strategy = "epoch",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        gradient_checkpointing=True,
    )

    # define multiple label metrics
    def multi_label_metrics(predictions, labels, threshold=0.5):

        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))

        # next, use threshold to turn them into integer predictions
        # if tuning, use the best threshold for each label
        y_pred = np.zeros(probs.shape)
        if args.tuning:
            for i in range(y_pred.shape[1]):
                prob = probs[:,i]
                label = labels[:,i]
                precision, recall, thresholds = precision_recall_curve(label, prob)
                f1 = 2 * (precision * recall) / (precision + recall)
                if np.isnan(f1).all():
                    threshold = 0.5
                else:
                    ix = np.nanargmax(f1)
                    threshold = thresholds[ix]
                y_pred[np.where(prob >= threshold),i] = 1
        else:
            y_pred[np.where(probs >= threshold)] = 1
        
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
        precision_micro_average = precision_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
        recall_micro_average = recall_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
        f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
        precision_macro_average = precision_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
        recall_macro_average = recall_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
        roc_auc_micro_average = roc_auc_score(y_true, probs, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        corr_s = corr_score(probs, y_true)

        # return as dictionary
        metrics = {
            'micro_f1': f1_micro_average,
            'micro_precision': precision_micro_average,
            'micro_recall': recall_micro_average, 
            'macro_f1': f1_macro_average,
            'macro_precision': precision_macro_average,
            'macro_recall': recall_macro_average,
            'micro_roc_auc': roc_auc_micro_average,
            'corr_score': corr_s,
            'accuracy': accuracy
            }
        
        # compute metrics for each label
        for idx in range(y_true.shape[1]):
            l = y_true[:,idx]
            p = y_pred[:,idx]
            name = id2label[idx]
            metrics[name + "_f1"] = f1_score(y_true=l, y_pred=p, zero_division=0)
            metrics[name + "_precision"] = precision_score(y_true=l, y_pred=p, zero_division=0)
            metrics[name + "_recall"] = recall_score(y_true=l, y_pred=p, zero_division=0)
        
        # load the lengths of the tasks
        with open("dataset_tmp/lengths.pkl", 'rb') as f:
            lengths = pkl.load(f)
        
        # compute metrics for each task
        start = 0
        for task in lengths:
            length = lengths[task]
            l_ = y_true[:, start:start+length]
            p_ = y_pred[:, start:start+length]
            metrics["task_" + task + "_micro_f1"] = f1_score(y_true=l_, y_pred=p_, average='micro', zero_division=0)
            metrics["task_" + task + "_micro_precision"] = precision_score(y_true=l_, y_pred=p_, average='micro', zero_division=0)
            metrics["task_" + task + "_micro_recall"] = recall_score(y_true=l_, y_pred=p_, average='micro', zero_division=0)
            metrics["task_" + task + "_macro_f1"] = f1_score(y_true=l_, y_pred=p_, average='macro', zero_division=0)
            metrics["task_" + task + "_macro_precision"] = precision_score(y_true=l_, y_pred=p_, average='macro', zero_division=0)
            metrics["task_" + task + "_macro_recall"] = recall_score(y_true=l_, y_pred=p_, average='macro', zero_division=0)

            start += length

        return metrics
    
    # wrapper for metric computation
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, 
                tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds, 
            labels=p.label_ids)
        return result

    # load the trainer
    trainer = Trainer(
        model,
        training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    if args.freeze == 'f':
        # freeze base
        for param in model.bert.parameters():
            param.requires_grad = False
    
    # train
    trainer.train()

    # evaluate on test set
    test_tuple = trainer.predict(encoded_dataset["test"])
    print("The evaluation metrics on the test set are:")
    print(test_tuple.metrics)
    
    torch.save(test_tuple, output_dir + "/" + "test_tuple.pt")
    torch.save(labels, output_dir + "/" + "labels.pt")
