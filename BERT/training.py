import argparse
import numpy as np
import pickle as pkl
from dataset_loading import loading_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, set_seed
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, roc_curve
import torch
import datasets
import wandb
import os

if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--top50', action='store_true')
    parser.add_argument('--tuning', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()

    # initialize wandb
    if int(os.environ["LOCAL_RANK"]) == 0:
        wandb.init(project="text2data", entity="olab", name=args.task + "(top50)" if args.top50 else args.task)
    
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
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")

    # preprocess for training
    def preprocess_data(examples):
        # take a batch of texts
        text = examples["TEXT"]
        # encode them
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
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
    model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT", 
                                                            problem_type="multi_label_classification", 
                                                            num_labels=len(labels),
                                                            id2label=id2label,
                                                            label2id=label2id)
    
    # set the training argument
    batch_size = 16
    metric_name = "f1"
    training_args = TrainingArguments(
        args.task + "-output(top50)" if args.top50 else args.task + "-output",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
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
                fpr, tpr, thresholds = roc_curve(label, prob)
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                threshold = thresholds[ix]
                y_pred[np.where(prob >= threshold),i] = 1
        else:
            y_pred[np.where(probs >= threshold)] = 1
        
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        precision_micro_average = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
        recall_micro_average = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, probs, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)

        # return as dictionary
        metrics = {
            'f1': f1_micro_average,
            'precision': precision_micro_average,
            'recall': recall_micro_average, 
            'roc_auc': roc_auc,
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
            metrics["task_" + task + "_f1"] = f1_score(y_true=l_, y_pred=p_, average='micro')
            metrics["task_" + task + "_precision"] = precision_score(y_true=l_, y_pred=p_, average='micro')
            metrics["task_" + task + "_recall"] = recall_score(y_true=l_, y_pred=p_, average='micro')

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

    if args.freeze:
        # freeze base
        for param in model.bert.parameters():
            param.requires_grad = False
    
    # train
    trainer.train()

    # evaluate on validation set
    eval_metrics = trainer.evaluate()
    print("The evaluation metrics on the validation set are:")
    print(eval_metrics)

    # evaluate on test set
    test_tuple = trainer.predict(encoded_dataset["test"])
    print("The evaluation metrics on the test set are:")
    print(test_tuple.metrics)
    if args.top50:
        torch.save(test_tuple, args.task + "-output(top50)/" + "test_tuple.pt")
        torch.save(labels, args.task + "-output(top50)/" + "labels.pt")
    else:
        torch.save(test_tuple, args.task + "-output/" + "test_tuple.pt")
        torch.save(labels, args.task + "-output/" + "labels.pt")
