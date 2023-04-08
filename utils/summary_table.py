import pandas as pd
import torch
import os

def summary(mtask, top50=False):
    """Print a summary table for a main task.

    Args:
        task (Task): Task to summarize.
        top50 (bool): If True, summarize the top 50 results. 
        If False, summarize the full results.
    """

    # construct a dataframe
    models = ["BERT", "RoBERTa", "Longformer"]
    corrloss = ["original", "CorrLoss"]
    index = pd.MultiIndex.from_product([models, corrloss])
    if mtask == "PROC":
        columns = ["PROC_ICD9", "PROC_ICD9+CPT", "PROC_ICD9+DRG", "PROC_ICD9+DIAG"]
    elif mtask == "DIAG":
        columns = ["DIAG_ICD9", "DIAG_ICD9+CPT", "DIAG_ICD9+DRG", "DIAG_ICD9+PROC"]    
    else:
        raise ValueError("Unknown task: %s" % mtask)
    
    # create an empty dataframe
    df = pd.DataFrame(index=index, columns=columns)

    # define postfix for the results
    postfix = "-output(top50)" if top50 else "-output"

    # fill the dataframe
    # for each model
    for model in models:
        if model == "BERT":
            model_name = "Bio_Discharge_Summary_BERT_n"
        elif model == "RoBERTa":
            model_name = "roberta-base_n"
        elif model == "Longformer":
            model_name = "longformer-base-4096_n"
        
        # for different version of loss functions
        for corr in corrloss:
            if corr == "CorrLoss":
                model_name = model_name + "_corrloss"

            # for each task
            for col in columns:
                main_task = col.split('+')[0]
                if '+' in col:
                    if col.endswith("CPT"):
                        task = main_task + "-CPT_CD"
                    elif col.endswith("DRG"):
                        task = main_task + "-DRG_CODE"
                    elif col.endswith("PROC"):
                        task = main_task + "-PROC_ICD9"
                    elif col.endswith("DIAG"):
                        task = main_task + "-DIAG_ICD9"
                    
                    task = task + postfix
                else:
                    task = col + postfix

                # read the results
                path = "../language_model/" + model_name + "/" + task + "/"
                if os.path.exists(path + "test_tuple.pt"):
                    # get the macro f1 score and fill the dataframe
                    result_tuple = torch.load(path + "test_tuple.pt")
                    result = result_tuple.metrics
                    macro_f1 = round(result["test_task_" + main_task + "_macro_f1"], 4)
                    df.loc[(model, corr), col] = macro_f1
                else:
                    continue
    
    if mtask == "PROC":
        df.columns = ["PROC", "PROC+CPT", "PROC+DRG", "PROC+DIAG"]
    elif mtask == "DIAG":
        df.columns = ["DIAG", "DIAG+CPT", "DIAG+DRG", "DIAG+PROC"]   
    return df

