import pandas as pd
import os
from sklearn.model_selection import train_test_split

seed = 1
path = '/gpfs/data/oermannlab/project_data/text2table/complete/'

# Load pre-concatenated input data: clinical notes in text format
df_input = pd.read_csv((path + 'complete_input_dataset.csv'), delimiter=',', index_col=[0])
df_input = df_input.reset_index(drop=True)

# Load pre-concatenated output data: clinical notes in table format
df_output = pd.read_csv((path + 'complete_output_dataset.csv'), delimiter=',', index_col=[0])
df_output = df_output.reset_index(drop=True)

# Compare whether pre-concatenated data have the same set of admission ID
# Input
x = df_input.HADM_ID.unique().tolist()
newListX = sorted(x)
# Output
y = df_output.HADM_ID.unique().tolist()
newListY = sorted(y)
# Compare
if newListX == newListY:
    print ("The lists are identical")
else :
    print ("The lists are not identical")

# Merge columns from pre-concatenated input data to pre-concatenated output data
df_input = df_input.sort_values(by=['HADM_ID']).set_index(['HADM_ID'])
df_output = df_output.sort_values(by=['HADM_ID'])
df_output = df_output.join(df_input, on=['HADM_ID'])

# Put the 'SUBJECT_ID' (patient ID) at the first column of the table
first_column = df_output.pop('SUBJECT_ID')
df_output.insert(0, 'SUBJECT_ID', first_column)

# Reset the index and ready for export
df_output = df_output.reset_index(drop=True)

# Export the table into the target directory
df_output.to_csv((path + 'complete_merged_final_dataset.csv'))
print('Merge completed!')


# Let's do the training and testing set split
targetDir = '/gpfs/data/oermannlab/project_data/text2table/complete/train_test_data'
os.mkdir(targetDir)

# Split the data into training and testing set using sklearn: Split the data based on unique admission ID
admID_train, admID_test = train_test_split(df_output.HADM_ID.unique(), test_size=0.2, random_state=seed)
admID_train, admID_dev = train_test_split(admID_train, test_size=0.25, random_state=seed)

# Get the corresponding data from the merged final dataset
df_train = df_output[df_output.HADM_ID.isin(admID_train)]
df_dev = df_output[df_output.HADM_ID.isin(admID_dev)]
df_test = df_output[df_output.HADM_ID.isin(admID_test)]

# Reset the index
df_train = df_train.reset_index(drop=True)
df_dev = df_dev.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Export the training and testing set into the target directory
targetDir = targetDir + "/"
df_train.to_csv((targetDir + 'train.csv'))
df_test.to_csv((targetDir + 'test.csv'))
df_dev.to_csv((targetDir + 'dev.csv'))
print('Training and testing set split completed!')
