from scipy.stats import entropy
import pickle as pkl
import numpy as np
import pandas as pd
import argparse
import os

# Shannon entropy of a single label
def Shannon_entropy(y_true):
    n = len(y_true)
    c_1 = sum(y_true)
    c_0 = n - c_1
    entropy_ = entropy([c_0/n, c_1/n])
    balance = entropy_ / np.log(2)
    return balance

# Shannon entropy of a task
def Shannon_entropy_task(y_df, task):
    y_task = y_df[task]
    dummies = y_task.str.get_dummies(sep=" <CEL> ")
    dict = {}
    for col in dummies.columns:
        dict[col] = Shannon_entropy(dummies[col])
    return dict

if __name__ == "__main__":
    # Because our model is trained on training set, we only calculate the balance of the training set
    # Load the training set
    data_dir="/gpfs/data/oermannlab/project_data/text2table/complete_v2/train_test_data/"
    train=pd.read_csv(data_dir+'/train.csv')

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()

    # calcualte the balance and save it to a pickle file
    balance = Shannon_entropy_task(train, args.task)
    if not os.path.exists("balance"):
        os.makedirs("balance")
    with open("balance/" + args.task + '-balance.pkl', 'wb') as f:
        pkl.dump(balance, f)
