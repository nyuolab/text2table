import pandas as pd
import os

# Concatenation for the Input data: Clinical Notes in the text format

# Patients info - ID file Path
patientInfo = '/gpfs/data/oermannlab/project_data/text2table/complete/ID.csv'
IDs = None

# Original MIMIC-3 dataset file path
path = '/gpfs/data/oermannlab/users/cz1906/mimic-iii-clinical-database-1.4/'
# Read the Input files
note = pd.read_csv(path + 'NOTEEVENTS.csv', delimiter=',', dtype=str, low_memory=False)

# If the Patients info - ID file exists, load it directly from the directory
# If else, extract the information here, and store it into the directory
if not (os.path.exists(patientInfo)):
    admission = pd.read_csv(path + 'ADMISSIONS.csv', delimiter=',', dtype=str, low_memory=False)
    IDs = admission[['SUBJECT_ID', 'HADM_ID']]
    IDs = IDs.sort_values(by=['SUBJECT_ID', 'HADM_ID'])
    IDs.to_csv(patientInfo)

else:
    IDs = pd.read_csv(patientInfo, delimiter=',', dtype=str)


# Merge the Patients ID with their corresponding clinicial notes
ID = IDs
new_note = note.sort_values(by=['SUBJECT_ID', 'HADM_ID']).set_index(['SUBJECT_ID', 'HADM_ID'])[['CATEGORY', 'TEXT']]
new_dataset = IDs.join(new_note, on=['SUBJECT_ID', 'HADM_ID'])

# Initialize the final datatset for the input data (COL1: SUBJECT_ID; COL2: HADM_ID; COL3: TEXT)
final_dataset = pd.DataFrame.from_dict({'SUBJECT_ID': [], 'HADM_ID': [], 'TEXT': []})

# Concatenate the notes for each unique patient
for patientID in (ID.SUBJECT_ID.unique()):
    # Individual unique patient
    tmpPatient = new_dataset.loc[new_dataset['SUBJECT_ID'] == patientID]
    # Individual unique Patient with his/her corresponding clinicial notes for multiple admissions
    tmpNote = ''

    # Concatenate the notes for cuurent patient
    for current in (tmpPatient['CATEGORY'].unique()):
        # Category for the current notes
        category = (str(current).strip())[:3].upper()
        # Notes under this category
        currentNotes = tmpPatient.loc[tmpPatient['CATEGORY'] == (str(current).strip())]['TEXT']
        
        adm = '<ADM> '
        # If the patient has multiple notes (ADMISSIONS) under this category
        if len(currentNotes) > 1:
            tmpNote += ('<' + category + '> ')
            s = ' <ADM> '.join([str(n) for n in currentNotes])
            tmpNote += adm
            tmpNote += s
            tmpNote += ' '

        # If the patient has only one notes under this category
        elif len(currentNotes) == 1:
            tmpNote += ('<' + category + '> ')
            tmpNote += adm
            tmpNote += str(currentNotes[currentNotes.index[0]])
            tmpNote += ' '

        # If the patient has no note under this category: Skip
        else:
            continue


    # Get the correspnding Patient ID and Admission ID
    tmpID = tmpPatient['SUBJECT_ID'].unique()
    tmpADM = tmpPatient['HADM_ID'].unique()

    # Concatenate multiple IDs together if necessary
    if len(tmpID) > 1:
        tmpID = ', '.join([str(n) for n in tmpID])
    else:
        tmpID = str(tmpID[0])

    if len(tmpADM) > 1:
        tmpADM = ', '.join([str(n) for n in tmpADM])
    else:
        tmpADM = str(tmpADM[0])

    # Store the current patient into the final dataset
    full_note = {'SUBJECT_ID': tmpID, 'HADM_ID': tmpADM, 'TEXT': tmpNote}
    final_dataset = pd.concat([final_dataset, pd.DataFrame.from_records([full_note])])


# Store the Final dataset into a .csv file
final_dataset.to_csv('/gpfs/data/oermannlab/project_data/text2table/complete/final_dataset_input.csv')
