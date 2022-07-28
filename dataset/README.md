Final Dataset for the Text2Table Task
========================================

The files for creating the final dataset for the Text2Table Task. The whole dataset is created based on the 
present of MIMIC-III (Medical Information Mart for Intensive Care III), and more details can be acccessed at
[MIMIC-III](https://mimic.mit.edu/docs/iii/). As the original MIMIC-III dataset
contains 26 tables and approximately hundreds of columns, some pre-selected columns used to train the model
can be accessed at `text2table/dataset/Pre-Selected_Columns.txt`. In addition, some sample clinicial notes
for patients can be accessed at `text2table/dataset/sample_patient_data/`. However, as we want models to 
produce more generalized and useful contents, we include as many as clinical notes and columns to feed the
model.

Requirements
------------------
Make sure to run `make requirements` before running the actual model. And you can also run
`make test_environment` to see you have met the requirments.

Input Dataset: Clinical Notes in text format
-----------------------------------------------
The table, `NOTEEVENTS.csv`, in the original MIMIC-III dataset, contains information about patient's clinical 
notes in differnt categories associated with each individual admission. Therefore, the input for the model, in the text format, 
is created based on this table. All clinical notes assoicated with each patient for the individual admission are concatenated 
together as pieces of data. Notice that there are differnt categories of clinical notes present for one admission. Therefore, 
one row presents one unique admission, and the differnt categories of notes are represented in different columns. (As there are categories
such as, 'Case Management', 'Pharmacy', 'Social Work', 'Rehab Services', 'Consult', presented in the original dataset; however, for our 
interest, we manually read those texts and conclude they are irrelevent to our prediction. Thus, we exclude those columns) 
To create the input dataset, one can run `python3 final_dataset_input.py`, and the resulting data contains 58976 
admissions associated with different patients for the corresponding clinical notes.

Output Dataset: Clinical Notes in the table format
-----------------------------------------------------
The rest of the tables in the original MIMIC-III dataset contains information about different patients for their corresponding
clinical notes in the table format. The output of the model, in the table format, is created based on those tables. All minimum
valuable columns that contain information about patients for their admissions are concatenated together to be the output for models.
To create the output dataset, one can run `python3 final_dataset_output.py`.

Final Dataset: All in one table
----------------------------------
To put clinical notes and assoicated information in one table, one can run `python3 final_dataset_merge.py`. This file helps to put all
clinical notes in both text format and table format at just one table, and indeed for feeding the model. Also, this program helps to 
split the train, development, and evaluation sets (80%/5%/15%) and store three sets into three files umder the same directory.

---------

To create the final dataset, please make sure to download the original MIMIC-III dataset and make it present in an accessible directory.