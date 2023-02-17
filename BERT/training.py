import argparse
import numpy as np
from dataset_loading import loading_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
import torch
import datasets

if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()

    # path
    data_dir="/gpfs/data/oermannlab/project_data/text2table/complete_v2/train_test_data/"

    # load the dataset
    [train, dev, test] = loading_dataset(data_dir, args.task.split('-'))
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
        args.task + "_output",
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
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        precision_micro_average = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
        recall_micro_average = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {
            'f1': f1_micro_average,
            'precision': precision_micro_average,
            'recall': recall_micro_average, 
            'roc_auc': roc_auc,
            'accuracy': accuracy
            }
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

    # freeze base
    for param in model.bert.parameters():
        param.requires_grad = False
    
    # train
    trainer.train()

    # evaluate
    trainer.evaluate()
