# This is a script to calculate the correlation between the main and auxilary tasks
# Specifically, it calculates the correlation between every label of the main task and the auxilary task as a whole

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import argparse
import pickle as pkl
import os

# Calculate the correlation between every label of the main task and the auxilary task as a whole
def task_corr(main, auxilary, train_df, top50=False):

    # Drop the rows with missing values
    train_df=train_df.dropna(how='all', subset=[main, auxilary])

    # get the dummy variables for the main and auxilary tasks
    main_train = train_df[main]
    aux_train = train_df[auxilary]
    dummy_main = main_train.str.get_dummies(sep=' <CEL> ')
    dummy_aux = aux_train.str.get_dummies(sep=' <CEL> ')

    # if top50 is True, only consider the top 50 labels of the main task and the auxilary task
    if top50:
        folder_name = '../language_model/roberta-base_n/' + main + '-' + auxilary + '-output(top50)'
        labels = torch.load(folder_name + '/labels.pt')

        # get the top 50 labels of the main task
        main_labels = []
        for label in labels:
            if label.startswith(main):
                main_labels.append(label.split('-')[-1])
        
        # get the top 50 labels of the auxilary task
        aux_labels = []
        for label in labels:
            if label.startswith(auxilary):
                aux_labels.append(label.split('-')[-1])
        
        # drop the labels that are not in the top 50 labels
        dummy_main = dummy_main[main_labels]
        dummy_aux = dummy_aux[aux_labels]


    # dataframe to store the correlation values
    corr_df = pd.DataFrame(columns=dummy_main.columns)

    # iterate through the labels of the main task
    for col in tqdm(dummy_main.columns):

        # store the correlation values for each label of the main task
        corr_list = []

        # iterate through the labels of the auxilary task
        for col2 in dummy_aux.columns:

            # calculate the correlation between the label of the main task and 
            # the label of the auxilary task
            # and take the absolute value of the correlation
            # we take the absolute value because 
            # we are only interested in the magnitude of the correlation
            # and the originla correlation coefficients can cancel out each other
            corr_val = abs(dummy_main[col].corr(dummy_aux[col2]))

            # append the correlation value to the list
            corr_list.append(corr_val)
        
        # calculate the mean of all the correlation values for the label of the main task 
        # and append it to the corresponding column in the dataframe
        corr_df[col] = [np.mean(corr_list)]
    
    return corr_df

# main function
if __name__ == "__main__":
    
    # Because our model is trained on training set, we only calculate the correlation on the training set
    # Load the training set
    data_dir="/gpfs/data/oermannlab/project_data/text2table/complete_v2/train_test_data/"
    train=pd.read_csv(data_dir+'/train.csv')

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', type=str, required=True)
    parser.add_argument('--aux', type=str, required=True)
    parser.add_argument('--top50', action='store_true')
    args = parser.parse_args()

    # calcualte the correlation and save it to a pickle file
    corr_df = task_corr(args.main, args.aux, train, args.top50)
    
    # create a directory to store the correlation dataframes
    if not os.path.exists('correlation'):
        os.makedirs('correlation')
    
    # save the correlation dataframe to a pickle file
    if args.top50:
        open("correlation/"+args.main+'-'+args.aux+'-corr(top50).pkl', 'wb').write(pkl.dumps(corr_df))
    else:
        open("correlation/"+args.main+'-'+args.aux+'-corr.pkl', 'wb').write(pkl.dumps(corr_df))

