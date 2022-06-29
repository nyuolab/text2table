import pandas as pd

path = '/gpfs/data/oermannlab/project_data/text2table/complete/'

# Load pre-concatenated input data: clinical notes in text format
df_input = pd.read_csv((path + 'final_dataset_input.csv'), delimiter=',', index_col=[0])
df_input = df_input.reset_index(drop=True)

# Load pre-concatenated output data: clinical notes in table format
df_output = pd.read_csv((path + 'complete_dataset.csv'), delimiter=',', index_col=[0])
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

# Merge columns 'TEXT' and 'SUBJECT_ID' from pre-concatenated input data to pre-concatenated output data
df_input = df_input.sort_values(by=['HADM_ID']).set_index(['HADM_ID'])[['TEXT', 'SUBJECT_ID']]
df_output = df_output.sort_values(by=['HADM_ID'])
df_output = df_output.join(df_input, on=['HADM_ID'])

# Put the 'SUBJECT_ID' (patient ID) at the first column of the table
first_column = df_output.pop('SUBJECT_ID')
df_output.insert(0, 'SUBJECT_ID', first_column)

# Reset the index and ready for export
df_output = df_output.reset_index(drop=True)

# Export the table into the target directory
targetDir = '/gpfs/data/oermannlab/project_data/text2table/final_dataset/'
df_output.to_csv((targetDir + 'final_dataset.csv'))
print('Merge completed!')
