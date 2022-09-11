#!/usr/bin/python
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

#!/usr/bin/python

def predict(X,y): #helper function for predict_train() and predict_test()
    #load xg-boost model
    with open(os.path.join(baseline_folder_path,model_path),'rb') as f:
        model=pkl.load(f)

    # predict
    y_pred_proba = model.predict_proba(X)
    #np.save(os.path.join(baseline_folder_path,'y_pred_proba.npy'),y_pred_proba)
    print("y_pred_proba: ",y_pred_proba.shape)
    print("num class assigned for 1st input: ",y[0].sum())
    auc=metrics.roc_auc_score(y, y_pred_proba, average="weighted", multi_class="ovr")
    
    y_pred= model.predict(X)
    
    f1=metrics.f1_score(y, y_pred, average="micro")
    acc=metrics.accuracy_score(y, y_pred)
    print("auc: ",auc)
    print("f1: ",f1)
    print("accuracy: ",acc)

def predict_train():
    print("Doing prediction on train data")
    print("load pre-processed X_train and y_train for evaluation.")
    # --change
    X_train=np.load(os.path.join(baseline_folder_path,'train',X_path))
    y_train=np.load(os.path.join(baseline_folder_path,'train',y_path))
    print("X_train's type: ",type(X_train))
    print("X_train's shape: ",X_train.shape)
    print("y_train's type: ",type(y_train))
    print("y_train's shape: ",y_train.shape)
    #call predict function
    predict(X_train,y_train)
    
def predict_test():
    # preprocess data
    if os.path.exists(os.path.join(baseline_folder_path,args.partition,X_path)) and os.path.exists(os.path.join(baseline_folder_path,args.partition,y_path)):
        print("load pre-existing preprocessed data.")
        X_test=np.load(os.path.join(baseline_folder_path,args.partition,X_path))
        y_test=np.load(os.path.join(baseline_folder_path,args.partition,y_path))
    else:
        # this code should be useless
        # just in case test data isn't already preprocessed
        print("preprocessing test data...")
        #os.makedirs(test_folder_path,exist_ok=True)
        #load original dataset
        test=pd.read_csv(os.path.join(data_dir,args.partition+'.csv'))
            
        #get rid of rows with nan in labels
        test=test[test['ICD9_CODE'].isna()==False]

        X_test=test['TEXT']
        y_test=test['ICD9_CODE']
        
        #load tokenizer
        with open(os.path.join(baseline_folder_path,tokenizer_save_path), 'rb') as f:
            tokenizer=pkl.load(f)
        X_test =tokenizer.transform(X_test)
        X_test=X_test.toarray()

        # transform y_train into dummy variables
        y_test=y_test.str.get_dummies(sep=' <CEL> ').to_numpy()

    print("X_test's type: ",type(X_test))
    print("X_test's shape: ",X_test.shape)
    print("y_test's type: ",type(y_test))
    print("y_test's shape: ",y_test.shape) 

    #call predict function
    predict(X_test,y_test)

def transform_save(X,y,part,tokenizer):
    X=tokenizer.transform(X)
    X=X.toarray()

    part_path=os.path.join(baseline_folder_path,part)
    os.makedirs(part_path,exist_ok=True)
    #save matrix X
    np.save(os.path.join(part_path,X_path),X)

    # save dummified y_train
    np.save(os.path.join(part_path,y_path),y)
    
    print(f"{part} X shape: ",X.shape)
    print(f"{part} y shape: ",y.shape)
    return X,y

def split(final):
    train, test = train_test_split(final, test_size=0.2, random_state=1)
    train, dev = train_test_split(train, test_size=0.25, random_state=1)
    return train,dev,test

def train():
    #preprocess data
    if os.path.exists(os.path.join(baseline_folder_path,args.mode,X_path)) and os.path.exists(os.path.join(baseline_folder_path,args.mode,y_path)):
        print("load pre-existing preprocessed data.")
        X_train=np.load(os.path.join(baseline_folder_path,args.mode,X_path))
        y_train=np.load(os.path.join(baseline_folder_path,args.mode,y_path))
        print(type(X_train))
        print(type(y_train))
    else:
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
        total=total[total['ICD9_CODE'].isna()==False]

        X_total=total['TEXT']
        y_total=total['ICD9_CODE']

        # tokenizer 
        if args.tokenizer=='bag_of_words':
            tokenizer= CountVectorizer(lowercase=True,max_features=1000,dtype=float)
        elif args.tokenizer=='tfidf':
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
        # transform y_train into dummy variables
        print("dumification...")
        y_total=y_total.str.get_dummies(sep=' <CEL> ').to_numpy()
        
        #--test

        print("splitting...")
        # split train, dev, test:
        X_train,X_dev,X_test=split(X_total)
        y_train,y_dev,y_test=split(y_total)   
        
        print("tokenizing...")
        # fit on train data
        tokenizer.fit(X_train)
        #save tokenizer
        with open(os.path.join(baseline_folder_path,tokenizer_save_path), 'wb') as f:
            pkl.dump(tokenizer, f)
        
        print("tokenizing finished")

        # transform X data
        X_train,y_train=transform_save(X_train,y_train,"train",tokenizer)
        X_dev,y_dev=transform_save(X_dev,y_dev,"dev",tokenizer)
        X_test,y_test=transform_save(X_test,y_test,"test",tokenizer)

    print(X_train.shape)
    print(y_train.shape)
    # create XGBoost instance with default hyper-parameters
    xgb_estimator = xgb.XGBClassifier(tree_method='hist',n_jobs=-1,n_estimators=10,verbosity=3)

    # fit the mod
    xgb_estimator.fit(X_train, y_train)
    #save model
    with open(os.path.join(baseline_folder_path,model_path),'wb') as f:
        pkl.dump(xgb_estimator,f)

# ========== main function ============

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer',help='select tokenizers: "bag_of_words", "tfidf" ',required=True)
parser.add_argument('--mode',help='select mode: "train", "predict_train","predict_test"',required=True)
parser.add_argument('--partition',help='dev or test',required=False)
args = parser.parse_args()

# arg_dict = {'tokenizer': args.tokenizer, 'mode': args.mode,'partition':args.partition}
# print(arg_dict)

# shared path vars across different modes
data_dir="/gpfs/data/oermannlab/project_data/text2table/minimum_re_adtime"
X_path='matrix_x.npy'
y_path='dum_y.npy'
tokenizer_save_path='baseline_tokenizer.json'
model_path='no_dask_xgb.pkl'

X_test_path='test_rep.npy'
y_test_path='dum_y_test.npy'

baseline_folder_path='baseline_folder/'+args.tokenizer
#test_folder_path='baseline_folder/'+arg_dict['tokenizer']+'test/'

print("baseline_folder_path: ",baseline_folder_path)

if args.mode=='predict_train':
    predict_train()
elif args.mode=='train':
    train()
elif args.mode=='predict_test':
    if args.partition is None:
        raise ValueError("Please specify 'partition' as 'dev' or 'test', which tells the model which partition of the full dataset you want to perform inference on")
    predict_test()

