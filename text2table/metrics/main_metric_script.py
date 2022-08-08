from datetime import date
from text2table.text2table.metrics.date_metric import *
from text2table.text2table.metrics.multilabel import *
from text2table.text2table.metrics.singlelabel import *
import evaluate

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class MainMetric(datasets.Metric):
    """TODO: Short description of my metric."""

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            
            # Note: we have to define the features as simply 'predictions' and 'reference' instead of our actuall dataset features (column names) which will lead to a ValueError. (Since huggingface will assume that the column name are the actual columns being passed into ref/pred, which isn't what we're doing here)
            
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            }),
            # Homepage of the metric for documentation
            homepage="http://metric.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"]
        )
    
    def cel_match(mode,cel_pred,cel_ref,result,category):
        # --unsure
        for c in mode:
            result[f'{c}_{curr_header}']['ele_total']+=len(cel_ref)
        #iterate thru each element in a cell
        for ele_pred, ele_ref in zip(cel_pred, cel_ref):
            


            char_right=0 # counts number of chars matching
            char_len=len(ele_ref) # counts number of charcters in this column cell
            for c,d in zip(ele_pred,ele_ref): #c and d are each char in word a,b
                if c==d: char_right+=1 #if c,d match, count as char_right
            #find the length of the longer element/word
            max_len=max(len(ele_pred),len(ele_ref))
            #find char_wrong
            char_wrong=max_len-char_right
            #--crucial: different config modes:
            for c in mode:
                perc=c/100
                if not (perc>=0 and perc<=1): raise ValueError(f"Invalid config name for ColMatch: {c}. Please input a valid number for percentage between 0 and 100 inclusive") 
                #if number of matching chars smaller than length of word
                if char_wrong/char_len<=perc: 
                    result[f'{c}_{curr_header}']['ele_match']+=1
        return result

    def _compute(self, predictions, references,inputs): #predictions, references both in a batch
        """Returns the scores"""
        # #log config:
        # os.makedirs('eval_logs',exist_ok=True)
        # date=datetime.datetime.now()
        # n=date.strftime("eval_logs/%m_%d_%H:%M:%S_eval.log")
    
        #iterate thru rows/inputs
        for row_pred,row_ref,row_input in zip(predictions, references,inputs):
            # get category
            category_token=row_input.split(' ')[0]

            # check single vs multi label
            if ' <CEL> ' in row_ref: # multi label
                if ' <CEL> ' not in row_pred: # error: prediction only has 1 label
                    print('error: supposed to be multi label, but prediction only generates single label.')
                    continue 
                # check categories under multi label
                if category_token in ['<DIAG_ICD9>','<PROC_ICD9>','<CPT_CD>']: # hierarchy
                    error=evaluate.load('multilabel.py').compute(predictions=row_pred,references=row_pred)

                    print()
            else: # single label
                # check categories under single label
                if category_token in ['<DOB>']: #time
                    error=evaluate.load('date_metric.py').compute(predictions=row_pred,references=row_pred)
                    print()
                elif category_token in ['<GENDER>','<HOSPITAL_EXPIRE_FLAG>']: # binary (singlelabel.py)
                    error=evaluate.load('singlelabel.py').compute(predictions=row_pred,references=row_pred)
                    print()
                elif 
                
        

            #check type of category 
            


            elif category_token in ['<PRESCRIPTION>','<DRG_CODE>']: # multi-char exact match (col_wise_metric_script.py)


            elif category_token in ['<LAB_MEASUREMENT>']:
                print()
