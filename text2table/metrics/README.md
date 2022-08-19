Model Evaluation Metrics
========================================

The folder contains all neccessary metrics for evaluating the model's performance.

The file, `main_metric_script.py`, is the main evaluation script. This file handles model's predictions and assigns
the proper evaluation metrics for different types of output, enabling the goal of multi-tasking. As the task is 
generation-based，special tokens will help to separate the different types of outputs.

The file, `singlelabel.py`, is the metric (F1, Accuracy, Precision) used to compute notes like gender, flags, etc. The 
file, `date_metric.py`, is the metric (how many days are off by the ground truth) used to compute the notes like 
admission date, DOB, etc. The file, `class_metric.py`, is the metric (F1) used to compute the notes like ICD-9 code,
DRG code, etc. Three files are used together to measure the model's performance by running the main evaluation script.

The file, `col_wise_metric_script.py`, is the main evalution script for MVP model. The evaluation is based on the percentage
wrong in the pridiction (0%, 10%, 20% wrong). Note, this file can only be used to evaluate GENDER, DOB, ADMITTIME, and ICD9.
