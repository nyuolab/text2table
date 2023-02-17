import pandas as pd
from datasets import Dataset
import pickle as pkl
import os

def loading_dataset(data_dir, task):
    train=pd.read_csv(data_dir+'/train.csv')
    dev=pd.read_csv(data_dir+'/dev.csv')
    test=pd.read_csv(data_dir+'/test.csv')

    # drop missing values
    train=train.dropna(how='all', subset=task)
    dev=dev.dropna(how='all', subset=task)
    test=test.dropna(how='all', subset=task)

    # get X and y based on the task
    X_train=train['TEXT'].reset_index(drop=True)
    y_train=train[task].astype('string')
    X_dev=dev['TEXT'].reset_index(drop=True)
    y_dev=dev[task].astype('string')
    X_test=test['TEXT'].reset_index(drop=True)
    y_test=test[task].astype('string')     

    # dummify classes
    print("dumification...")

    y_list = [y_train, y_dev, y_test]

    # outer join to sync the columns
    y_total = pd.concat(y_list).reset_index(drop=True)

    # The list of dummies of tasks
    dummies = []

    # stores the number of labels of each task
    lengths = {}

    # stores the name of labels of each task
    names = {}

    # loop through the list of tasks
    for t in task:

        # dummify the labels
        dummified = y_total[t].str.get_dummies(sep=" <CEL> ")

        # add the prefix of the task name to the columns
        dummified = dummified.add_prefix(t + "-")
        
        # add the names of labels
        names[t] = dummified.columns

        # add the length of the corresponding task
        lengths[t] = dummified.shape[1]

        # print the shape of the task we are dummifying
        print("the shape of " + t + ": ", dummified.shape)

        # add the dummified labels to the dummy list
        dummies.append(dummified)
        
    # concatenate the dummified labels
    y_total = pd.concat(dummies, axis=1)

    # create a folder to store the lengths and names if it does not exist
    if not os.path.exists('dataset_tmp'):
        os.makedirs('dataset_tmp')

    # save the lengths of all the tasks to a file
    with open(os.path.join('dataset_tmp', 'lengths.pkl'),'wb') as f:
        pkl.dump(lengths,f)

    # save the names to a file
    with open(os.path.join('dataset_tmp', 'names.pkl'), 'wb') as f:
        pkl.dump(names,f)
    
    for i, y in enumerate(y_list):
        y_list[i] = y_total[:y.shape[0]].reset_index(drop=True).copy()
        y_total = y_total.drop(y_total.index[:y.shape[0]])
    
    # split the dummified labels
    y_train = y_list[0]
    y_dev = y_list[1]
    y_test = y_list[2]

    print("the shape of y_train: ", y_train.shape)
    print("the shape of X_train: ", X_train.shape)
    print("the shape of y_dev: ", y_dev.shape)
    print("the shape of X_dev: ", X_dev.shape)
    print("the shape of y_test: ", y_test.shape)
    print("the shape of X_test: ", X_test.shape)

    print("dumification finished")

    # concatenate the dummified labels with the texts
    train_ = pd.concat([X_train, y_train], axis=1)
    dev_ = pd.concat([X_dev, y_dev], axis=1)
    test_ = pd.concat([X_test, y_test], axis=1)

    print("the shape of train_", train_.shape)
    print("the shape of dev_", dev_.shape)
    print("the shape of test_", test_.shape)

    # load the dataset from pandas dataframe
    train_dataset = Dataset.from_pandas(train_, preserve_index=False)
    dev_dataset = Dataset.from_pandas(dev_, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_, preserve_index=False)

    print("the shape of train_dataset", train_dataset.shape)
    print("the shape of dev_dataset", dev_dataset.shape)
    print("the shape of test_dataset", test_dataset.shape)

    print("dataset loaded")

    return train_dataset, dev_dataset, test_dataset

        
    
    
