from datetime import date
from text2table.metrics.exact_match import get_wrong_char
import evaluate
import pickle as pkl
import os
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


    def _compute(self, predictions, references,inputs): #predictions, references both in a batch
        """Returns the scores"""
        # #log config:
        # os.makedirs('eval_logs',exist_ok=True)
        # date=datetime.datetime.now()
        # n=date.strftime("eval_logs/%m_%d_%H:%M:%S_eval.log")
        
        #initiate dictionary
        result={}
        
        for i in [f'<{x}>' for x in ['DIAG_ICD9', 'PROC_ICD9', 'PRESCRIPTION', 'DRG_CODE','LAB_MEASUREMENT','HOSPITAL_EXPIRE_FLAG', 'GENDER', 'CPT_CD','DOB']]:
            result[i]['pred']=[]
            result[i]['ref']=[]

        # initiate ways to record errors
        result['label_mismatch']=0

        #iterate thru rows/inputs
        for row_pred,row_ref,row_input in zip(predictions, references,inputs):
            if (' <CEL> ' in row_ref) and (' <CEL> ' not in row_pred): # error if ref is multi label but prediction only has 1 label
                print('error: supposed to be multi label, but prediction only generates single label.')
                    result['label_mismatch']+=1
                continue 

            # get category
            category_token=row_input.split(" ")[0]
        
            # for each row of data: append data to different keys of the dictionary based on category
            result[category_token]['pred']+=row_pred
            result[category_token]['ref']+=row_ref

        # Evaluation: calculation of metrics (batch-wise)
        final={}
        # DOB:
        final['<DOB>']=evaluate.load('date_metric.py').compute(predictions=result['<DOB>']['pred'],references=result['<DOB>']['ref'])
        
        # binary:
        for category_token in ['<GENDER>','<HOSPITAL_EXPIRE_FLAG>']:
            final[category_token]=evaluate.load('singlelabel.py').compute(predictions=result[category_token]['pred'],references=result[category_token]['ref'])

        # classes
        # class_file: a file that consists of all the unique classes of a certain category, well be passed to the 'class_metric.py''s compute function
        for category_token in ['<DIAG_ICD9>','<PROC_ICD9>','<CPT_CD>','<PRESCRIPTION>','<DRG_CODE>']: 
            if category_token=='<DIAG_ICD9>':
                class_file=os.path.join('class_files','diag_icd_classes.pkl')
            elif category_token=='<PROC_ICD9>':
                class_file=os.path.join('class_files','proc_icd_classes.pkl')
            elif category_token=='<CPT_CD>':
                class_file=os.path.join('class_files','cpt_classes.pkl')
            elif category_token=='<PRESCRIPTION>':
                class_file=os.path.join('class_files','pres_classes.pkl')
            elif category_token=='<DRG_CODE>':
                class_file=os.path.join('class_files','drg_classes.pkl')

            final[category_token]=evaluate.load('class_metric.py').compute(predictions=result[category_token]['pred'],references=result[category_token]['ref'],classfile=class_file)

        return final




