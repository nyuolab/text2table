import pandas as pd
import dask.dataframe as dd

# load dataset(all the neccessary charts)
path = "/gpfs/data/oermannlab/public_data/MIMIC/mimic-iii-clinical-database-1.4/"
admission = dd.read_csv(path + "ADMISSIONS.csv", dtype = "str")
drgcodes = dd.read_csv(path + "DRGCODES.csv", dtype = "str")
patient = dd.read_csv(path + "PATIENTS.csv", dtype = "str")
procedure_icd = dd.read_csv(path + "PROCEDURES_ICD.csv", dtype = "str")
cpt = dd.read_csv(path + "CPTEVENTS.csv", dtype = "str")
diagnose_icd = dd.read_csv(path + "DIAGNOSES_ICD.csv", dtype = "str")
note = pd.read_csv(path + "NOTEEVENTS.csv", dtype = "str")

# extract the subject ID and the admission ID from the admission chart
# subject id is for joining patients' personal information including gender and DOB
# admission id is for joining all the other information
ID = admission[["SUBJECT_ID", "HADM_ID", "HOSPITAL_EXPIRE_FLAG"]]

# extract all the columns that we selected for the final dataset
# set their indexes to IDs for joining them later
patient_ = patient[["SUBJECT_ID", "GENDER", "DOB"]].set_index("SUBJECT_ID")
patient_["DOB"] = patient_["DOB"].apply(lambda x: x[:10])
cpt_ = cpt.set_index(["HADM_ID"])[["CPT_CD"]]
drgcodes_ = drgcodes.set_index(["HADM_ID"])[["DRG_CODE"]]
diagnose_icd_ = diagnose_icd.set_index(["HADM_ID"])[["ICD9_CODE"]]
diagnose_icd_.columns = ["DIAG_ICD9"]
procedure_icd_ = procedure_icd.set_index(["HADM_ID"])[["ICD9_CODE"]]
procedure_icd_.columns = ["PROC_ICD9"]
note_ = note[note["CATEGORY"] == "Discharge summary"].set_index(["HADM_ID"])[["TEXT"]]
note_ = dd.from_pandas(note_, npartitions = 900)

# function for merging all the rows that have the same admission ID
# desgined to be used in pandas groupby.apply()
# take in a dataframe that should be a group chunk resulting from groupby() 
# return a dataframe with only one row that contains all the information associated with the corresponding admission ID
# join all the cells in the same columns with ' <CEL> ' as the delimiter
def merge_row(df):
    dict_merged = {}
    for colname in df.columns:
        dict_merged[colname] = [' <CEL> '.join(df[colname].dropna().astype(str).sort_values().unique())]
    df_merged = pd.DataFrame(dict_merged)
    return df_merged

# This is just another version of the above function with " <NTE> " as the delimiter
# Due to the difference between pandas dataframe and dask dataframe,
# I created this function instead of adding a argument to the above function
# to avoid any potential error
def merge_row_nte(df):
    dict_merged = {}
    for colname in df.columns:
        dict_merged[colname] = [' <NTE> '.join(df[colname].dropna().astype(str).sort_values().unique())]
    df_merged = pd.DataFrame(dict_merged)
    return df_merged

# for all the code below, new_dataset is a dask dataframe and new_dataset_ is a pandas dataframe
# join GENDER, DOB, CPT_CD, and DRG_CODE
new_dataset = ID.join(patient_, on="SUBJECT_ID")[["HADM_ID", "GENDER", "DOB", "HOSPITAL_EXPIRE_FLAG"]]
new_dataset = new_dataset.join(cpt_, on="HADM_ID")
new_dataset = new_dataset.join(drgcodes_, on="HADM_ID")

# execute all the tasks assigned to the dask dataframe and convert it to a pandas dataframe
new_dataset_ = new_dataset.compute()

# group rows by their admission IDs and merge all rows in each group chunk
new_dataset_ = new_dataset_.groupby("HADM_ID", group_keys = False).apply(merge_row)
new_dataset_ = new_dataset_.reset_index(drop = True)

# convert the pandas dataframe to a dask dataframe
# join DIAG_ICD9
new_dataset = dd.from_pandas(new_dataset_, npartitions = 900)
new_dataset = new_dataset.join(diagnose_icd_, on="HADM_ID")

# execute all the tasks assigned to the dask dataframe and convert it to a pandas dataframe
new_dataset_ = new_dataset.compute()

# group rows by their admission IDs and merge all rows in each group chunk
new_dataset_ = new_dataset_.groupby("HADM_ID", group_keys = False).apply(merge_row)
new_dataset_ = new_dataset_.reset_index(drop = True)

# convert the pandas dataframe to a dask dataframe
# join PROC_ICD9
new_dataset = dd.from_pandas(new_dataset_, npartitions = 900)
new_dataset = new_dataset.join(procedure_icd_, on="HADM_ID")

# execute all the tasks assigned to the dask dataframe and convert it to a pandas dataframe
new_dataset_ = new_dataset.compute()

# group rows by their admission IDs and merge all rows in each group chunk
new_dataset_ = new_dataset_.groupby("HADM_ID", group_keys = False).apply(merge_row)
new_dataset_ = new_dataset_.reset_index(drop = True)

# convert the pandas dataframe to a dask dataframe
# join TEXT
new_dataset = dd.from_pandas(new_dataset_, npartitions = 900)
new_dataset = new_dataset.join(note_, on="HADM_ID")

# execute all the tasks assigned to the dask dataframe and convert it to a pandas dataframe
new_dataset_ = new_dataset.compute()

# group rows by their admission IDs and merge all rows in each group chunk
new_dataset_ = new_dataset_.groupby("HADM_ID", group_keys = False).apply(merge_row_nte)
new_dataset_ = new_dataset_.reset_index(drop = True)

# drop the rows that have missing values in TEXT column
new_dataset_ = new_dataset_.dropna(subset = "TEXT")

# save the result to a .csv file
new_dataset_.to_csv("/gpfs/data/oermannlab/project_data/text2table/complete_v2/dataset_v2.csv", index = False)
