# This is a script to calculate the correlation between the main and auxilary tasks
# Specifically, it calculates the correlation between every label of the main task and the auxilary task as a whole

import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import pickle as pkl

# Calculate the correlation between every label of the main task and the auxilary task as a whole
def task_corr(main, auxilary, train_df):

    # Drop the rows with missing values
    train_df=train_df.dropna(how='all', subset=[main, auxilary])

    # get the dummy variables for the main and auxilary tasks
    main_train = train_df[main]
    aux_train = train_df[auxilary]
    dummy_main = main_train.str.get_dummies(sep=' <CEL> ')
    dummy_aux = aux_train.str.get_dummies(sep=' <CEL> ')

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
            corr_val = dummy_main[col].corr(dummy_aux[col2])

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
    args = parser.parse_args()

    # calcualte the correlation and save it to a pickle file
    corr_df = task_corr(args.main, args.aux, train)
    open("correlation/"+args.main+'-'+args.aux+'-corr.pkl', 'wb').write(pkl.dumps(corr_df))

