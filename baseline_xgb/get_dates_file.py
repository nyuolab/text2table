import numpy as np
import pandas as pd
import pickle as pkl
import os

total=pd.read_csv('/gpfs/data/oermannlab/project_data/text2table/complete_v2/dataset_v2.csv')

def tmp(row):
    item=row.split('-')
    item[0]='Y'+item[0]
    item[1]='M'+item[1]
    item[2]='D'+item[2]
    return '-'.join(item)
total['DOB']=total['DOB'].apply(lambda x:tmp(x))

dum=total['DOB'].str.get_dummies(sep='-')

# just in case folder doesn't exist
class_dir='class_files'
os.makedirs(class_dir,exist_ok=True)

# save dumifies columns
file_name='dates.pkl'
with open(os.path.join(class_dir,file_name),'wb') as f:
    pkl.dump(dum.columns,f)