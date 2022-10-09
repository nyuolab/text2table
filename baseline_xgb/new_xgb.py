from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle as pkl
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn import metrics
import argparse
from sklearn.model_selection import train_test_split
from more_itertools import powerset
import shutil


# Code for Training and Testing the baseline model, XGBoost, on the data with entered task
#
# The following is the guide for runing the code
# To play around with this code, plesase run the following command:
# python3 new_xgb.py --all=<...> --tokenizer=<...> --mode=<...> --task=<...> --partition=<...>
#
# all and tokenizer are required.
# If you want to run all combinations of tasks, please enter --all=y
# if not, please enter --all=n and specify what task you want to run through the rest of arguments
# For tokenizer, you can choose from 'bag_of_words ' or 'tfidf'
# For mode, you can choose from 'train' or 'predict_train' or 'predict_test'
# For task, you can choose a branch of tasks, including "gender", "DOB", "expire_flag", "cpt_cd", "drg_code", "icd9_dia", "icd9_proc"
# (Note: you can enter multiple tasks at the same time, but you will need to use "/" to separate them. For example, to run gender and DOB, you
# should enter "gender/DOB")
# For partition, you can choose from 'dev' or 'test' (note: you can only choose 'dev' or 'test' when mode is 'predict_test')
#
# Here is some example input for the code
# For all experiments
# python3 new_xgb.py --all=y --tokenizer=bag_of_words
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

    # load the dictionary that stores length of each task
    with open(os.path.join(baseline_folder_path, lengths_path), 'rb') as f:
        lengths = pkl.load(f)
    
    # predict on X to get the AUC score
    y_pred_proba = model.predict_proba(X)

    # stores all the metric numbers for each task
    results = {}

    # loop through lengths to separate the tasks and get the metric results for each task
    for k in lengths:

        # stores the results of each metric
        metric = {}

        # get the part for the task and delete it from the array
        y_ = y[:, :lengths[k]]
        y = np.delete(y, np.s_[:lengths[k]], 1)
        y_pred_proba_ = y_pred_proba[:, :lengths[k]]
        y_pred_proba = np.delete(y_pred_proba, np.s_[:lengths[k]], 1)
        # y_pred_ = y_pred[:, :lengths[k]]
        # y_pred = np.delete(y_pred, np.s_[:lengths[k]], 1)

        # prepare the y_true and y_pred_proba for micro-average roc curve
        y_flat = y_.ravel()
        y_pred_proba_flat = y_pred_proba_.ravel()

        # calculate gmeans to find out the optimal threshold
        fpr, tpr, thresholds = metrics.roc_curve(y_flat, y_pred_proba_flat)
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        best_threshold = thresholds[ix]
        print("the best threshold is: ", best_threshold)

        # convert all probabilities into decisions
        y_pred_ = np.where(y_pred_proba_ >= best_threshold, 1, 0)
        metric["threshold"] = best_threshold

        # micro and macro auc
        try:
            metric["auc_micro"] = metrics.roc_auc_score(y_, y_pred_proba_, average="micro", multi_class="ovo")
            metric["auc_macro"] = metrics.roc_auc_score(y_, y_pred_proba_, average="macro", multi_class="ovo")
        except ValueError:
            print("Only one class present in y_true. ROC AUC score is not defined in that case.")
            
        # micro and macro f1
        metric["f1_micro"] = metrics.f1_score(y_, y_pred_, average="micro")
        metric["f1_macro"] = metrics.f1_score(y_, y_pred_, average="macro")

        # micro and macro precision
        metric["precision_micro"] = metrics.precision_score(y_, y_pred_, average="micro")
        metric["precision_macro"] = metrics.precision_score(y_, y_pred_, average="macro")

        # micro and macro recall
        metric["recall_micro"] = metrics.recall_score(y_, y_pred_, average="micro")
        metric["recall_macro"] = metrics.recall_score(y_, y_pred_, average="macro")

        # accuracy
        metric["accuracy"] = metrics.accuracy_score(y_, y_pred_)
        
        # add all the results
        results[k] = metric
    
    # display results and store it into a .pkl file
    print(results)
    os.makedirs("results_" + args.tokenizer,exist_ok=True)
    result_file_name = "_".join([x.replace("_", "-") for x in lengths.keys()]) + ".pkl"
    with open(os.path.join("results_" + args.tokenizer, result_file_name), 'wb') as f:
        pkl.dump(results,f)
        

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
        # load original dataset
        train=pd.read_csv(data_dir+'/train.csv')
        dev=pd.read_csv(data_dir+'/dev.csv')
        test=pd.read_csv(data_dir+'/test.csv')

        # recover the original dataframe
        total=pd.concat([train, dev])
        total=pd.concat([total, test])
        print("The shape of the whole dataset: ", total.shape)
        # get rid of rows that consist of missing values only
        total=total.dropna(how='all', subset=task)

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

        # The list of dummies of tasks
        dummies = []

        # stores the number of labels of each task
        lengths = {}

        # loop through the list of tasks
        for t in task:

            # if the task is DOB, labels are dummified with - as separator
            if "DOB" == t:
                dummified = y_total[t].str.get_dummies(sep='-').to_numpy()
            
            # if not, labels are dummified with <CEL> as separator
            else:
                dummified = y_total[t].str.get_dummies(sep=" <CEL> ").to_numpy()

            # print the shape of the task we are dummifying
            print("the shape of " + t + ": ", dummified.shape)

            # add the dummified labels to the dummy list
            dummies.append(dummified)

            # add the length of the corresponding task
            lengths[t] = dummified.shape[1]
        
        # concatenate the dummified labels
        y_total = np.concatenate(dummies, axis=1)

        # print the shape of all tasks concatnated together
        print("the shape of tasks of interest after concatenation: ", y_total.shape)

        # save the lengths of all the tasks to a file
        with open(os.path.join(baseline_folder_path, lengths_path),'wb') as f:
            pkl.dump(lengths,f)   
        
        print("dumification finished")


        print("splitting...")
        # split
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
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', help='whether run all experiments: y, n', required=True)
    parser.add_argument('--tokenizer',help='select tokenizers: "bag_of_words", "tfidf" ',required=True)
    parser.add_argument('--mode',help='select mode: "train", "predict_train","predict_test"',required=False)
    parser.add_argument('--task', help='select task: "gender", "DOB", "expire_flag", "cpt_cd", "drg_code", "icd9_dia", "icd9_proc"', required=False)
    parser.add_argument('--partition',help='dev or test',required=False)
    args = parser.parse_args()

    # shared path vars across different modes
    data_dir="/gpfs/data/oermannlab/project_data/text2table/complete_v2/train_test_data/"
    X_path='matrix_x.npy'
    y_path='dum_y.npy'
    tokenizer_save_path='baseline_tokenizer.json'
    model_path='no_dask_xgb.pkl'
    lengths_path='lengths.pkl'

    X_test_path='test_rep.npy'
    y_test_path='dum_y_test.npy'

    baseline_folder_path='baseline_folder/'+args.tokenizer
    print("Save results to baseline_folder_path: ", baseline_folder_path)

    if args.all == "y":
        # all tasks
        tasks = ["GENDER", "HOSPITAL_EXPIRE_FLAG", "CPT_CD", "DRG_CODE", "DIAG_ICD9", "PROC_ICD9"]

        # generate powerset of all tasks
        tasks_comb = [list(x) for x in list(powerset(tasks))[1:]]

        # run all experiments
        for task in tasks_comb:

            # train xgb
            args.mode = "train"
            train(task, args.tokenizer)

            # get results on test set
            args.partition = "test"
            predict_test("test")

            # remove the model to prepare for next iteration
            shutil.rmtree("baseline_folder")
    
    else:
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
            predict_test(args.partition)

        else: # invalid mode
            print("Invalid mode. Please choose from train, predict_train, predict_test")
            exit()
