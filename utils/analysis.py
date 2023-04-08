import pandas as pd
import pickle as pkl
import torch

def scenario_analysis(mtask, top50=False, corrloss=False, correlation=False):

    # construct a dataframe
    models = ["BERT", "RoBERTa", "Longformer"]
    columns = ["top50", "bottom50"]
    if mtask == "PROC_ICD9":
        auxs = ["+CPT", "+DRG", "+DIAG"]
    elif mtask == "DIAG_ICD9":
        auxs = ["+CPT", "+DRG", "+PROC"]    
    else:
        raise ValueError("Unknown task: %s" % mtask)
    
    index = pd.MultiIndex.from_product([models, auxs])
    
    # create an empty dataframe
    df = pd.DataFrame(index=index, columns=columns)

    # define postfix for the results
    postfix = "-output(top50)" if top50 else "-output"

    # define postfix for model name
    postfix_model = "_corrloss" if corrloss else ""

    # fill the dataframe
    # for each model
    for model in models:
        if model == "BERT":
            model_name = "Bio_Discharge_Summary_BERT_n"
        elif model == "RoBERTa":
            model_name = "roberta-base_n"
        elif model == "Longformer":
            model_name = "longformer-base-4096_n"

        model_name = model_name + postfix_model

        # for each task
        for aux in auxs:
            # get the task folder name
            if aux == "+CPT":
                task_ = mtask + "-CPT_CD"
            elif aux == "+DRG":
                task_ = mtask + "-DRG_CODE"
            elif aux == "+DIAG":
                task_ = mtask + "-DIAG_ICD9"
            elif aux == "+PROC":
                task_ = mtask + "-PROC_ICD9"
            
            task = task_ + postfix

            # the path to the results with and without auxillary task
            aux_path = "../language_model/" + model_name + "/" + task + "/"
            path = "../language_model/" + model_name + "/" + mtask + postfix + "/"

            # load the results
            output_aux = torch.load(aux_path + "test_tuple.pt")

            output = torch.load(path + "test_tuple.pt")

            # get the label balances
            with open("../utils/balance/" + mtask + "-balance.pkl", "rb") as f:
                balances = pkl.load(f)
            
            # get the correlation between each label of the main task and the auxillary task as a whole
            # if top50 is true, then the results are from the top50
            if top50:
                with open("../utils/correlation/" + task_ + "-corr(top50).pkl", "rb") as f:
                    correlations = pkl.load(f)
            else:
                with open("../utils/correlation/" + task_ + "-corr.pkl", "rb") as f:
                    correlations = pkl.load(f)
            
            # get the metrics of the two outputs
            metrics = output.metrics
            metrics_aux = output_aux.metrics

            # get the list of labels that have non-zero performance from the results without auxillary task
            l = []
            for metric in metrics:
                if ('test_' + mtask) in metric and metrics[metric] != 0:
                    l.append(metric)
            
            # get the list of labels that have non-zero performance from the results with auxillary task
            l_aux = []
            for metric in metrics_aux:
                if ('test_' + mtask) in metric and metrics_aux[metric] != 0:
                    l_aux.append(metric)
            
            # get the intersection of the two lists
            list_intersection = list(set(l).intersection(set(l_aux)))

            # filter the labels that are not in the label balances
            list_intersection = [label for label in list_intersection if label.split('-')[-1].split('_')[0] in balances.keys()]

            # filter the labels that are not in the correlations
            list_intersection = [label for label in list_intersection if label.split('-')[-1].split('_')[0] in correlations.columns]

            if correlation:
                correlations = correlations[[label.split('-')[-1].split('_')[0] for label in list_intersection]]
                correlations = correlations.to_dict('list')
                
                # convert the value of the dict from a list to a value
                correlations = {key: value[0] for key, value in correlations.items()}

                # sort the correlations by the value in descending order
                sorted_dict = {k: v for k, v in sorted(correlations.items(), key=lambda item: item[1], reverse=True)}

            else:
                # filter out the balances that are not in the list_intersection
                balances = {key: value for key, value in balances.items() if key in [label.split('-')[-1].split('_')[0] for label in list_intersection]}

                # sort balances by the the value in descending order
                sorted_dict = {k: v for k, v in sorted(balances.items(), key=lambda item: item[1], reverse=True)}

            # get the top50 labels
            top50_labels = list(sorted_dict.keys())[:50]
            top50_metrics = [metric for metric in list_intersection if metric.split('-')[-1].split('_')[0] in top50_labels]

            # get the difference between the main task without and with auxillary task for the top 50 labels
            top50_diff = [metrics_aux[metric] - metrics[metric] for metric in top50_metrics]

            # get the percentage of the differences that are positive
            top50_pos = len([diff for diff in top50_diff if diff > 0]) / len(top50_diff)

            # fill in the dataframe
            df.loc[(model, aux), "top50"] = round(top50_pos, 3)

            # get the bottom50 labels
            bottom50_labels = list(sorted_dict.keys())[-50:]
            bottom50_metrics = [metric for metric in list_intersection if metric.split('-')[-1].split('_')[0] in bottom50_labels]

            # get the difference between the main task without and with auxillary task for the bottom 50 labels
            bottom50_diff = [metrics_aux[metric] - metrics[metric] for metric in bottom50_metrics]

            # get the percentage of the differences that are positive
            bottom50_pos = len([diff for diff in bottom50_diff if diff > 0]) / len(bottom50_diff)

            # fill in the dataframe
            df.loc[(model, aux), "bottom50"] = round(bottom50_pos, 3)
        
    return df



            