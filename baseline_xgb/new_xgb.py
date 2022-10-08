import sys
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle as pkl
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn import metrics
import argparse
from sklearn.model_selection import train_test_split
from get_dates_file import tmp


# Code for Training and Testing the baseline model, XGBoost, on the data with entered task
#
# The following is the guide for runing the code
# To play around with this code, plesase run the following command:
# python3 new_xgb.py --tokenizer=<...> --mode=<...> --task=<...> --partition=<...>
#
# For tokenizer, you can choose from 'bag_of_words ' or 'tfidf'
# For mode, you can choose from 'train' or 'predict_train' or 'predict_test'
# For task, you can choose a branch of tasks, including "gender", "DOB", "expire_flag", "cpt_cd", "drg_code", "icd9_dia", "icd9_proc"
# (Note: you can enter multiple tasks at the same time, but you will need to use "/" to separate them. For example, to run gender and DOB, you
# should enter "gender/DOB")
# For partition, you can choose from 'dev' or 'test' (note: you can only choose 'dev' or 'test' when mode is 'predict_test')
#
# Here is some example input for the code
# For train
# python3 new_xgb.py --tokenizer=bag_of_words --mode=train --task=gender
# python3 new_xgb.py --tokenizer=bag_of_words --mode=train --task=gender/icd9_dia/expire_flag
# For predict on train
# python3 new_xgb.py --tokenizer=bag_of_words --mode=predict_train --task=gender
# python3 new_xgb.py --tokenizer=bag_of_words --mode=predict_train --task=gender/icd9_dia/expire_flag
# For predict on test/dev
# python3 new_xgb.py --tokenizer=bag_of_words --mode=predict_test --task=gender --partition=test
# python3 new_xgb.py --tokenizer=bag_of_words --mode=predict_test --task=gender --partition=dev
# python3 new_xgb.py --tokenizer=bag_of_words --mode=predict_test --task=gender/icd9_dia/expire_flag --partition=test


# ========== Functions for Train and Test ============
def predict(X,y): # helper function for predict_train() and predict_test()
    # load trained xg-boost model
    with open(os.path.join(baseline_folder_path,model_path),'rb') as f:
        model=pkl.load(f)

    # predict on X to get the AUC score
    y_pred_proba = model.predict_proba(X)
    print("y_pred_proba: ", y_pred_proba.shape)
    print("num class assigned for 1st input: ", y[0].sum())
    auc_weighted = metrics.roc_auc_score(y, y_pred_proba, average="weighted", multi_class="ovr")
    auc_micro = metrics.roc_auc_score(y, y_pred_proba, average="micro", multi_class="ovo")
    auc_macro = metrics.roc_auc_score(y, y_pred_proba, average="macro", multi_class="ovo")
    auc = {"auc_weighted": auc_weighted, "auc_micro": auc_micro, "auc_macro": auc_macro} # AUC
    print("auc: ", auc)

    # Predict on X to get the results of remaining metrics
    y_pred= model.predict(X)
    f1_micro=metrics.f1_score(y, y_pred, average="micro")
    f1_macro=metrics.f1_score(y, y_pred, average="macro")
    f1 = {"f1_micro": f1_micro, "f1_macro": f1_macro} # F1
    print("f1: ", f1)
    precision_weight=metrics.precision_score(y, y_pred, average="weighted")
    precision_micro=metrics.precision_score(y, y_pred, average="micro")
    precision_macro=metrics.precision_score(y, y_pred, average="macro")
    precision = {"precision_weight": precision_weight, "precision_micro": precision_micro, "precision_macro": precision_macro} # Precision
    print("precision: ", precision)
    recall_weighted=metrics.recall_score(y, y_pred, average="weighted")
    recall_micro=metrics.recall_score(y, y_pred, average="micro")
    recall_macro=metrics.recall_score(y, y_pred, average="macro")
    recall = {"recall_weighted": recall_weighted, "recall_micro": recall_micro, "recall_macro": recall_macro} # Recall
    print("recall: ", recall)
    accuracy=metrics.accuracy_score(y, y_pred) # Accuracy
    print("accuracy: ", accuracy)


def predict_train(): # function to predict on train data

    print("Doing prediction on train data")
    print("load pre-processed X_train and y_train for evaluation.")

    # Load preprocessed trained data 
    X_train=np.load(os.path.join(baseline_folder_path,'train',X_path))
    y_train=np.load(os.path.join(baseline_folder_path,'train',y_path))
    print("X_train's type: ",type(X_train))
    print("X_train's shape: ",X_train.shape)
    print("y_train's type: ",type(y_train))
    print("y_train's shape: ",y_train.shape)
    # call predict function to get the results
    predict(X_train,y_train)
    

def predict_test(partition): # function to predict on dev/test data
    # Load preprocessed data
    if os.path.exists(os.path.join(baseline_folder_path,args.partition,X_path)) and os.path.exists(os.path.join(baseline_folder_path,args.partition,y_path)):
        print("load pre-existing preprocessed data.")
        X_test=np.load(os.path.join(baseline_folder_path,args.partition,X_path))
        y_test=np.load(os.path.join(baseline_folder_path,args.partition,y_path))

    else:
        print("Entered dev/test partition does not exist. Please run the script with --mode=train to preprocess the data and train model first.")
        exit()
        # # this code should be useless
        # # just in case test data isn't already preprocessed
        # print("preprocessing test data...")
        # #os.makedirs(test_folder_path,exist_ok=True)
        # #load original dataset
        # test=pd.read_csv(os.path.join(data_dir,args.partition+'.csv'))        
        # # get rid of rows with nan in labels
        # # test=test[test['DRG_CODE'].isna()==False]
        # # get X and y
        # X_test=test['TEXT']
        # y_test=test['HOSPITAL_EXPIRE_FLAG'].astype('string')
        # # load tokenizer and transform data for prediction
        # with open(os.path.join(baseline_folder_path,tokenizer_save_path), 'rb') as f:
        #     tokenizer=pkl.load(f)
        # X_test =tokenizer.transform(X_test)
        # X_test=X_test.toarray()
        # # transform y_train into dummy variables
        # y_test=y_test.str.get_dummies(sep=' <CEL> ').to_numpy()

    print("X_test's type: ",type(X_test))
    print("X_test's shape: ",X_test.shape)
    print("y_test's type: ",type(y_test))
    print("y_test's shape: ",y_test.shape) 

    # call predict function to get the results
    predict(X_test,y_test)


def transform_save(X,y,part,tokenizer): # save the preprocessed data for future use
    X=tokenizer.transform(X)
    X=X.toarray()

    part_path=os.path.join(baseline_folder_path,part)
    os.makedirs(part_path,exist_ok=True)
    # save matrix X
    np.save(os.path.join(part_path,X_path),X)

    # save dummified y_train
    np.save(os.path.join(part_path,y_path),y)
    
    print(f"{part} X shape: ",X.shape)
    print(f"{part} y shape: ",y.shape)
    return X,y # return X and y for training 


def split(final): # Function to split the data into train/val/test sets (70/10/20)
    random_state = 100
    train, test = train_test_split(final, test_size=0.2, shuffle=True, random_state=random_state)
    train, dev = train_test_split(train, test_size=0.125, shuffle=True, random_state=random_state)

    return train,dev,test


def train(task, tokenizer): # Function to train the model
    # load data directly if already preprocessed
    if os.path.exists(os.path.join(baseline_folder_path,args.mode,X_path)) and os.path.exists(os.path.join(baseline_folder_path,args.mode,y_path)):
        print("load pre-existing preprocessed data.")
        X_train=np.load(os.path.join(baseline_folder_path,args.mode,X_path))
        y_train=np.load(os.path.join(baseline_folder_path,args.mode,y_path))
        print(type(X_train))
        print(type(y_train))

    else: # preprocess data
        print("preprocessing data...")
        os.makedirs(baseline_folder_path,exist_ok=True)
        #load original dataset
        train=pd.read_csv(data_dir+'/train.csv')
        dev=pd.read_csv(data_dir+'/dev.csv')
        test=pd.read_csv(data_dir+'/test.csv')

        #append dataframe
        total=pd.concat([train,dev])
        total=pd.concat([total,test])
        print("total shape: ",total.shape)
        #get rid of rows with nan in labels
        total=total[total[col].isna()==False]

        # get X and y based on the task
        X_total=total['TEXT']
        y_total=total[task].astype('string')

        # Tokenize the text based on the input tokenizer
        if tokenizer=='bag_of_words':
            tokenizer= CountVectorizer(lowercase=True,max_features=1000,dtype=float)
        elif tokenizer=='tfidf':
            tokenizer=TfidfVectorizer(ngram_range=(1,1), 
                                    max_features=500,
                                    sublinear_tf=True, 
                                    strip_accents='unicode', 
                                    analyzer='word', 
                                    token_pattern="\w{1,}", 
                                    stop_words="english",
                                    max_df=0.95,
                                    min_df=2,
                                    lowercase=True,dtype=float)

        # dummify classes
        print("dumification...")
        if len(task) == 1: # Single task with one column
            if task[0] == "GENDER" or "HOSPITAL_EXPIRE_FLAG	": # Gender and Expire Flag are binary
                y_total=y_total.squeeze(axis=1).str.get_dummies().to_numpy()
            elif task[0] == "DOB": # DOB has format of YYYY-MM-DD
                # add special token before y/m/d
                y_total=y_total.apply(lambda x:tmp(x))
                y_total=y_total.str.get_dummies(sep='-').to_numpy()
            else: # other tasks are separated by <CEL>
                y_total=y_total.squeeze(axis=1).str.get_dummies(sep=' <CEL> ').to_numpy()

        else: # Multi-task: Combine all columns into one column and each column is separated by <CEL>
            if "DOB" in task: # DOB has format of YYYY-MM-DD, processed it first
                # Replace "-" in DOB with "<CEL>"
                y_total["DOB"] = y_total["DOB"].str.replace("-", "<CEL>")

            # Combine all columns into one column and each column is separated by <CEL>
            y_total=y_total.apply(lambda x: ' <CEL> '.join(str(x)), axis=1)
            y_total=y_total.str.get_dummies(sep=' <CEL> ').to_numpy()
        print("dumification finished")


        print("splitting...")
        # Cross validation split
        X_train, X_dev, X_test = split(X_total)
        y_train, y_dev, y_test = split(y_total)   
        print("splitting finished")
        
        print("tokenizing...")
        # fit on train data
        tokenizer.fit(X_train)
        # save tokenizer for later use
        with open(os.path.join(baseline_folder_path,tokenizer_save_path), 'wb') as f:
            pkl.dump(tokenizer, f)
        print("tokenizing finished")

        print("transforming and saving...")
        # transform X data
        X_train, y_train = transform_save(X_train, y_train, "train", tokenizer)
        X_dev, y_dev = transform_save(X_dev, y_dev, "dev", tokenizer)
        X_test, y_test = transform_save(X_test, y_test, "test", tokenizer)
        print("transforming and saving finished")

    # train model
    print("Test the input and output shapes of the model:")
    print(X_train.shape)
    print(y_train.shape)

    # create XGBoost instance with default hyper-parameters
    xgb_estimator = xgb.XGBClassifier(tree_method='hist',n_jobs=-1,n_estimators=10,verbosity=3)

    # fit the model
    xgb_estimator.fit(X_train, y_train)
    # save model
    with open(os.path.join(baseline_folder_path,model_path),'wb') as f:
        pkl.dump(xgb_estimator,f)



# ========== main function ============

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer',help='select tokenizers: "bag_of_words", "tfidf" ',required=True)
parser.add_argument('--mode',help='select mode: "train", "predict_train","predict_test"',required=True)
parser.add_argument('--task', help='select task: "gender", "DOB", "expire_flag", "cpt_cd", "drg_code", "icd9_dia", "icd9_proc"', required=True)
parser.add_argument('--partition',help='dev or test',required=False)
args = parser.parse_args()

# Test the arguments
# arg_dict = {'tokenizer': args.tokenizer, 'mode': args.mode, 'task': args.task, 'partition':args.partition}
# print(arg_dict)

# shared path vars across different modes
data_dir="/gpfs/data/oermannlab/project_data/text2table/complete_v2/train_test_data/"
X_path='matrix_x.npy'
y_path='dum_y.npy'
tokenizer_save_path='baseline_tokenizer.json'
model_path='no_dask_xgb.pkl'

X_test_path='test_rep.npy'
y_test_path='dum_y_test.npy'

baseline_folder_path='baseline_folder/'+args.tokenizer
print("Save results to baseline_folder_path: ", baseline_folder_path)


# Identify the task
task = args.task.split('/')
# List all the available tasks as a dictionary for easy access
tasks = {"gender":"GENDER", "DOB":"DOB", "expire_flag":"HOSPITAL_EXPIRE_FLAG", "cpt_cd":"CPT_CD", "drg_code":"DRG_CODE", "icd9_dia":"DIAG_ICD9", "icd9_proc":"PROC_ICD9"}

# Check if the task is valid
if len(task) == 1: # if there is only one task
    task = task[0]
    # Convert it to the format that is used in the dataset
    if task in tasks:
        task = [tasks[task]]
    else:
        print("Invalid task. Please choose from the following: gender, DOB, expire_flag, cpt_cd, drg_code, icd9_dia, icd9_proc")
        exit()

else: # if there are multiple tasks
    # Convert it to the format that is used in the dataset
    for i in range(len(task)):
        if task[i] in tasks:
            task[i] = tasks[task[i]]
        else:
            print("Invalid task. Please choose from gender, DOB, expire_flag, cpt_cd, drg_code, icd9_dia, icd9_proc")
            exit()
    
# Check if the mode is valid
if args.mode=='predict_train': # predict on train data
    predict_train()

elif args.mode=='train': # train the model
    train(task, args.tokenizer)

elif args.mode=='predict_test': # predict on test data
    if args.partition is None: # partition is required here
        raise ValueError("Please specify 'partition' as 'dev' or 'test', which tells the model which partition of the full dataset you want to perform inference on")
        exit()
    predict_test(args.partition)

else: # invalid mode
    print("Invalid mode. Please choose from train, predict_train, predict_test")
    exit()