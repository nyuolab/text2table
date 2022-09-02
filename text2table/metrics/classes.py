import numpy as np
import pandas as pd
import pickle as pkl
import os

data_dir = "/gpfs/data/oermannlab/project_data/text2table/complete/train_test_data"

#load original dataset
train=pd.read_csv(data_dir+'/train.csv')
dev=pd.read_csv(data_dir+'/dev.csv')
test=pd.read_csv(data_dir+'/test.csv')
#append dataframe
total=pd.concat([train,dev])
total=pd.concat([total,test])

fold='text2table/text2table/metrics/class_files'

def gen_class_file(total,col_name,file_name):
    a=total[total.category==col_name]['label']

    dum=a.str.get_dummies(sep=' <CEL> ')
    
    with open(os.path.join(fold,file_name),'wb') as f:
        pkl.dump(dum.columns,f)

gen_class_file(total,'PRESCRIPTION','pres_classes.pkl')
gen_class_file(total,'DRG_CODE','drg_classes.pkl')
gen_class_file(total,'CPT_CD','cpt_classes.pkl')
gen_class_file(total,'DIAG_ICD9','diag_icd_classes.pkl')
gen_class_file(total,'PROC_ICD9','proc_icd_classes.pkl')