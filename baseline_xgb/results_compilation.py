import pandas as pd
import pickle as pkl
import os

# script for compiling all the results into a single csv

# The directory that stores all the results
result_directory = "results_bag_of_words"

# a dataframe with all the columns
result_df = pd.DataFrame(columns=["threshold", "auc_micro", "auc_macro", "f1_micro", 
"f1_macro", "precision_micro", "precision_macro", "recall_micro", 
"recall_macro", "accuracy"])

# a list of tuple for creating multiindex
result_index = []

# loop through files in the directory
for filename in os.listdir(result_directory):
    fn = os.path.join(result_directory, filename)
    # checking if it is a file
    if os.path.isfile(fn):
        with open(fn, 'rb') as f:
            # load the result dictionary from the file
            result = pkl.load(f)
            # append each result into the dataframe and the corresponding multiindex into the list
            for category in result:
                result_index.append((filename[:-4], category))
                result_df = pd.concat([result_df, pd.DataFrame([result[category]])])

# create multiindex from the list of tuples
index = pd.MultiIndex.from_tuples(result_index, names=["task", "category"])

# append the multiindex to the dataframe
result_df.index = index

# sort the dataframe according to the index
result_df = result_df.sort_index()

# display the dataframe and store the dataframe into a csv
print(result_df)
result_df.to_csv("results.csv")