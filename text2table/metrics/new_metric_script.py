# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""

import datasets
import datetime
import os
from text2table.logging_utils.logging_script import setup_logger
import evaluate
import pickle as pkl

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:metric,
title = {A great new metric},
authors={huggingface, Inc.},
year={2020}
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
calculate 0 wrong and percentage wrong with single-value as a unit
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> my_new_metric = datasets.load_metric("my_new_metric")
    >>> results = my_new_metric.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""

#helper for ColMatch's compute function: calculates matches within a cell (works for both cells with single or mult values)

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ColMatch(datasets.Metric):
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
        
    def _compute(self, predictions, references,mode): #predictions, references both in a batch
        """Returns the scores"""
        #log config:
        os.makedirs('eval_logs',exist_ok=True)
        date=datetime.datetime.now()
        n=date.strftime("eval_logs/%m_%d_%H:%M:%S_eval.log")
       
        metric_logger = setup_logger(name='null_logger', log_file=n,formatter='%(levelname)s:%(message)s')

        metric_logger.info('\n---------Start of evaluation epoch---------')

        #get column header from first reference
        headers=references[0].split(' <ROW> ')[0].split(' <COL> ')
        headers[0]=headers[0].replace('<s>','')
        metric_logger.info('headers: '+','.join(headers))

        #initiate dictionary
        result={}
        for i in headers:
            result[i]={}
            result[i]['pred']=[]
            result[i]['ref']=[]
            # result[i]={'pred':[]}
            # result[i]={'ref':[]}
        #can't find <row> separator
        result['<row>_error']=0
        #unequal number of columns
        result['<col>_mismatch']=0
        #--debug variable for metric_logger
        count=0

        #iterate thru rows/inputs and append to result
        for row_pred,row_ref in zip(predictions, references):
            #--debug variable for metric_logger
            count+=1
            metric_logger.info(f'\ncurrent row in batch: {count}')

            #split pred_str by columns as a list
            #replace <pad> since annoying
            row_ref=row_ref.replace('<pad>','')
            row_pred=row_pred.replace('<pad>','')
            metric_logger.info(f'row_ref: {row_ref}')
            metric_logger.info(f'row_pred: {row_pred}')

            metric_logger.info(f'start result: {result}')
            #row error:
            if ' <ROW> ' not in row_pred: 
                result['<row>_error']+=1
                metric_logger.info('<row>_error detected')
                metric_logger.info(f'result: {result}')
                continue

            cols_pred=row_pred.split(' <ROW> ')[1].split(' <COL> ')
            cols_ref=row_ref.split(' <ROW> ')[1].split(' <COL> ')
            metric_logger.info(f"cols_pred: {cols_pred}")
            metric_logger.info(f"cols_ref: {cols_ref}")

            #if length mismatch, log as error
            if len(cols_pred)!=len(cols_ref):
                result['<col>_mismatch']+=1
                metric_logger.info('<col>_mismatch detected')
                metric_logger.info(f'result: {result}')
                continue
 
            for i in range(len(headers)):
                # print("i: ",i)
                # print('current header: ',headers[i])
                # print('pred: ',pred)
                # print(result)
                if headers[i]=='ICD9_CODE':
                    # print('cols_pred: ',cols_pred)
                    # print('no split pred: ',cols_pred[i])
                    # print('no split ref: ',cols_ref[i])
                    # print('ICD9_CODE pred: ',cols_pred[i].split('</s>'))
                    # print('ICD9_CODE ref: ',cols_ref[i].split('</s>'))
                    pred_item=cols_pred[i].split('</s>')[0]
                    ref_item=cols_ref[i].split('</s>')[0]
                else:
                    pred_item=cols_pred[i]
                    ref_item=cols_ref[i]

                # --test
                print('pred: ',pred_item)
                print('ref: ',ref_item)

                result[headers[i]]['pred'].append(pred_item)
                result[headers[i]]['ref'].append(ref_item)
                metric_logger.info(f'result: {result}')
                

        # --debug
        # export icd_9 results (result[ICD9_CODE]) to folder icd9_debug
        pred_path='new_pred.pkl'
        ref_path='new_ref.pkl'
        debug_path='../metrics/icd9_debug'
        metric_path='new_metric'
        # make metric folder
        print('current path: ',os.getcwd())
        os.makedirs(os.path.join(debug_path,metric_path),exist_ok=True)
        # save both pred and ref
        with open(os.path.join(debug_path,metric_path,pred_path),'wb') as f:
            pkl.dump(result['ICD9_CODE']['pred'],f)
        with open(os.path.join(debug_path,metric_path,ref_path),'wb') as f:
            pkl.dump(result['ICD9_CODE']['ref'],f)

        # Evaluation: calculation of metrics (batch-wise)
        final={}

        # SEX
        category_token='SEX'
        #print('pred type:',type(result[category_token]['pred']))
        final[category_token]=evaluate.load('../metrics/singlelabel.py').compute(predictions=result[category_token]['pred'],references=result[category_token]['ref'])

        # DOB, ADMITTIME
        for category_token in ['DOB','ADMITTIME']:
            final[category_token]=evaluate.load('../metrics/date_metric.py').compute(predictions=result[category_token]['pred'],references=result[category_token]['ref'])
       
        # icd9
        category_token='ICD9_CODE'
        class_file=os.path.join('../metrics/class_files','diag_icd_classes.pkl')
        final[category_token]=evaluate.load('../metrics/class_metric.py').compute(predictions=result[category_token]['pred'],references=result[category_token]['ref'],classfile=class_file)

        metric_logger.info(f'final: {final}')
        metric_logger.info('\n---------End of evaluation epoch---------')
        
        # --debug
        # save final dictionary
        with open(os.path.join(debug_path,metric_path,'final.pkl'),'wb') as f:
            pkl.dump(final,f)

        return final