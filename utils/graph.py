import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import os

# plot the micro performance of the given task
def micro_performance_plot(dir, task, top50=False):

    # get the list of all directories
    dir_list = os.listdir(dir)
    dir_list = [d for d in dir_list if d.startswith(task)]
    dir_list = [d for d in dir_list if d.endswith("(top50)" if top50 else "output")]

    # dataframe to store the results
    df = pd.DataFrame(columns=["task", "metric", "value"])

    # iterate over all directories
    # and load the results to the dataframe
    for d in dir_list:
        output = torch.load(os.path.join(dir, d, "test_tuple.pt"))
        metrics = output.metrics
        for metric in ["f1", "precision", "recall"]:
            row = pd.DataFrame({"task": [' '.join(d.split('-')[:-1])], "metric": [metric], "value": [metrics["test_task_" + task + "_" + metric]]})
            df = pd.concat([df, row], ignore_index=True)
    
    # plot the results
    sns.set_theme(style="dark")
    fig = plt.figure(figsize=(20, 8))
    ax = sns.barplot(data=df, x="task", y="value", hue="metric", order=sorted(list(df["task"].unique())))
    ax.tick_params(axis='x', which='both', labelsize=15)
    ax.set(ylabel=None)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', label_type='edge', fontsize=15)
    if top50:
        plt.title("Micro performance of " + task + " (top50)", fontsize=15)
    else:
        plt.title("Micro performance of " + task, fontsize=15)
    ax.grid(axis='y')
    plt.legend(fontsize=15)
    plt.show()

# plot the performance change of each label of the given task
def label_performance_change(dir, b_dir, c_dir, task, aux, top50=False):

    # if top50 is true, then the results are from the top50
    postfix = "-output(top50)" if top50 else "-output"

    # get the output without auxillary task
    output = torch.load(os.path.join(dir, task + postfix, "test_tuple.pt"))

    # get the output with auxillary task
    output_aux = torch.load(os.path.join(dir, '-'.join([task, aux]) + postfix , "test_tuple.pt"))

    # get the label balances
    with open(os.path.join(b_dir, task + "-balance.pkl"), "rb") as f:
        balances = pkl.load(f)

    # get the correlation between each label of the main task and the auxillary task as a whole
    # if top50 is true, then the results are from the top50
    if top50:
        with open(os.path.join(c_dir, task + "-" + aux + "-corr(top50).pkl"), "rb") as f:
            correlations = pkl.load(f)
    else:
        with open(os.path.join(c_dir, task + "-" + aux + "-corr.pkl"), "rb") as f:
            correlations = pkl.load(f)
    
    # get the metrics of the two outputs
    metrics = output.metrics
    metrics_aux = output_aux.metrics

    # get the list of labels that have non-zero performance from the results without auxillary task
    l = []
    for metric in metrics:
        if ('test_' + task) in metric and metrics[metric] != 0:
            l.append(metric)
    
    # get the list of labels that have non-zero performance from the results with auxillary task
    l_aux = []
    for metric in metrics_aux:
        if ('test_' + task) in metric and metrics_aux[metric] != 0:
            l_aux.append(metric)
    
    # get the intersection of the two lists
    list_intersection = list(set(l).intersection(set(l_aux)))

    # filter the labels that are not in the label balances
    list_intersection = [label for label in list_intersection if label.split('-')[-1].split('_')[0] in balances.keys()]

    # filter the labels that are not in the correlations
    list_intersection = [label for label in list_intersection if label.split('-')[-1].split('_')[0] in correlations.columns]

    # get the labels that are f1, precision, and recall
    f1_labels = [label for label in list_intersection if "f1" in label]
    precision_labels = [label for label in list_intersection if "precision" in label]
    recall_labels = [label for label in list_intersection if "recall" in label]

    # create a dataframe for f1
    f1_df = pd.DataFrame(columns=["label", "diff", "balance", "correlation"])

    # iterate over the labels and add the the label name, difference of f1, the label balance, and the correlation
    for label in f1_labels:
        l = label.split('-')[-1].split('_')[0]
        row = pd.DataFrame({"label": [l], "diff": [metrics[label] - metrics_aux[label]], "balance": [balances[l]], "correlation": [correlations[l][0]]})
        f1_df = pd.concat([f1_df, row], ignore_index=True)
    
    # create a dataframe for precision
    precision_df = pd.DataFrame(columns=["label", "diff", "balance", "correlation"])

    # iterate over the labels and add the the label name, difference of precision, the label balance, and the correlation
    for label in precision_labels:
        l = label.split('-')[-1].split('_')[0]
        row = pd.DataFrame({"label": [l], "diff": [metrics[label] - metrics_aux[label]], "balance": [balances[l]], "correlation": [correlations[l][0]]})
        precision_df = pd.concat([precision_df, row], ignore_index=True)
    
    # create a dataframe for recall
    recall_df = pd.DataFrame(columns=["label", "diff", "balance", "correlation"])

    # iterate over the labels and add the the label name, difference of recall, the label balance, and the correlation
    for label in recall_labels:
        l = label.split('-')[-1].split('_')[0]
        row = pd.DataFrame({"label": [l], "diff": [metrics[label] - metrics_aux[label]], "balance": [balances[l]], "correlation": [correlations[l][0]]})
        recall_df = pd.concat([recall_df, row], ignore_index=True)
    
    # plot the results
    sns.set_theme(style="darkgrid")
    sns.set_context('talk')
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
    sns.scatterplot(ax = axes[0], data=f1_df, x="balance", y="diff", hue="correlation", size="correlation")
    axes[0].title.set_text("F1")
    sns.scatterplot(ax = axes[1], data=precision_df, x="balance", y="diff", hue="correlation", size="correlation")
    axes[1].title.set_text("Precision")
    sns.scatterplot(ax = axes[2], data=recall_df, x="balance", y="diff", hue="correlation", size="correlation")
    axes[2].title.set_text("Recall")
    postfix_ = "(top50)" if top50 else ""
    fig.suptitle("Performance change of each label of " + task + postfix_ + " with " + aux + postfix_, fontsize=15)
    plt.show()


    
    




