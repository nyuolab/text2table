import pandas as pd
import os
import numpy as np

# Concatenation for the Input data: Clinical Notes in the text format

# Patients info - ID file Path
patientInfo = '/gpfs/data/oermannlab/project_data/text2table/complete/ID.csv'
IDs = None

# Original MIMIC-3 dataset file path
path = '/gpfs/data/oermannlab/public_data/MIMIC/mimic-iii-clinical-database-1.4/'
# Read the Input files
note = pd.read_csv((path + 'NOTEEVENTS.csv'), delimiter=',', dtype=str, low_memory=False)

# If the Patients info - ID file exists, load it directly from the directory
# If else, extract the information here, and store it into the directory
if not (os.path.exists(patientInfo)):
    admission = pd.read_csv((path + 'ADMISSIONS.csv'), delimiter=',', dtype=str, low_memory=False)
    IDs = admission[['SUBJECT_ID', 'HADM_ID']]
    IDs = IDs.sort_values(by=['SUBJECT_ID', 'HADM_ID'])
    IDs = IDs.reset_index(drop=True)
    IDs.to_csv(patientInfo)

else:
    IDs = pd.read_csv(patientInfo, delimiter=',', dtype=str)


# Merge the Patients ID with their corresponding clinicial notes
ID = IDs
new_note = note.sort_values(by=['SUBJECT_ID', 'HADM_ID']).set_index(['SUBJECT_ID', 'HADM_ID'])[['CATEGORY', 'TEXT']]
new_dataset = IDs.join(new_note, on=['SUBJECT_ID', 'HADM_ID'])

# Get the info of all the unique categories of the clinical notes
categories = new_dataset['CATEGORY'].unique().tolist()
# Clear the space in the categories
categories = [str(x).strip() for x in categories]
# Remove the unwanted categories
unwanted = ['Case Management', 'Pharmacy', 'Social Work', 'Rehab Services', 'Consult', 'nan']
categories = [x for x in categories if x not in unwanted]

# Initialize the final datatset for the input data (COL1: SUBJECT_ID; COL2: HADM_ID; COL3-COL#: Categories of the clinical notes)
final_dataset = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID'] + categories)
# Input all unique HADM IDs and SUBJECT_IDs into the final dataset
final_dataset['SUBJECT_ID'] = ID['SUBJECT_ID']
final_dataset['HADM_ID'] = ID['HADM_ID']

# Concatenate the notes for each unique admission
for admID in (ID.HADM_ID.unique()):
    # Individual unique admission
    tmpADM = new_dataset.loc[new_dataset['HADM_ID'] == admID]
    # Unique categories of this admission
    tmpCat = tmpADM['CATEGORY'].unique().tolist()
    # Clear the space in the categories
    tmpCat = [str(x).strip() for x in tmpCat]
    # Remove the unwanted categories
    tmpCat = [x for x in tmpCat if x not in unwanted]

    # Concatenate the notes for current admission
    for current in tmpCat:
        # Individual unique admission for its corresponding unique clinicial note (initialization)
        tmpNote = ''
        # Category for the current notes
        category = (str(current).strip())[:3].upper()
        # Notes under this category
        currentNotes = tmpADM.loc[tmpADM['CATEGORY'] == (str(current).strip())]['TEXT']
        
        nte = '<NTE> '
        # If the current admission has multiple notes under this category
        if len(currentNotes) > 1:
            tmpNote += ('<' + category + '> ')
            s = ' <NTE> '.join([str(n) for n in currentNotes])
            tmpNote += nte
            tmpNote += s
            tmpNote += ' '

        # If the admission has only one note under this category
        elif len(currentNotes) == 1:
            tmpNote += ('<' + category + '> ')
            tmpNote += nte
            tmpNote += str(currentNotes[currentNotes.index[0]])
            tmpNote += ' '

        # If the admission has no notes under this category: Assign the empty string
        else:
            tmpNote = np.nan

        # Append the current notes for this category to the final dataset
        final_dataset.loc[final_dataset['HADM_ID'] == admID, (str(current).strip())] = tmpNote


# Replace the empty string with the NaN
final_dataset = final_dataset.replace(r'^\s*$', np.nan, regex=True)

# Clear all unwanted columns generated during concatenation
final_dataset = final_dataset[['SUBJECT_ID', 'HADM_ID'] + categories]

# Sort the row by HADM_ID and reset the index
final_dataset = final_dataset.sort_values(by=['HADM_ID'])
final_dataset = final_dataset.reset_index(drop=True)

# Store the Final dataset into a .csv file
final_dataset.to_csv('/gpfs/data/oermannlab/project_data/text2table/complete/complete_input_dataset.csv', index=False)
print("Input dataset is ready!")