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
def cel_match(mode,curr_header,cel_pred,cel_ref,result):
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
       
        metric_logger = setup_logger(name='metric_logger', log_file=n,formatter='%(levelname)s:%(message)s')

        metric_logger.info('\n---------Start of evaluation epoch---------')

        #get column header from first reference
        headers=references[0].split(' <ROW> ')[0].split(' <COL> ')
        headers[0]=headers[0].replace('<s>','')
        metric_logger.info('headers: '+','.join(headers))
        #initiate dictionary
        result={}
        for col in headers:
            for c in mode:
                result[f'{c}_{col}']={'ele_match':0,'ele_total':0}
        #can't find <row> separator
        result['<row>_error']=0
        #unequal number of columns
        result['<col>_mismatch']=0
        #--debug variable for metric_logger
        count=0
        #iterate thru rows/inputs
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

            #row error:
            if ' <ROW> ' not in row_pred: 
                result['<row>_error']+=1
                metric_logger.info('<row>_error detected')
                metric_logger.info(f'result: {result}')
                continue

            cols_pred=row_pred.split(' <ROW> ')[1].split(' <COL> ')
            cols_ref=row_ref.split(' <ROW> ')[1].split(' <COL> ')
            metric_logger.info(f"cols_pred: {', '.join(cols_pred)}")
            metric_logger.info(f"cols_ref: {', '.join(cols_ref)}")

            #if length mismatch, log as error
            if len(cols_pred)!=len(cols_ref):
                result['<col>_mismatch']+=1
                metric_logger.info('<col>_mismatch detected')
                metric_logger.info(f'result: {result}')
                continue

            # now for 1 row, iterate thru the columns
            for i in range(len(headers)):  
                metric_logger.info(f'current header: {headers[i]}')
                if ' <CEL> ' in cols_ref[i]: # use ref to be safe, if ref cell has multiple elements
                    metric_logger.info('This cell has multi values')
                    try: #if last column doesn't exist (a consequence of mismatch length of columns)
                        cel_pred=cols_pred[i].split(' <CEL> ')
                    except IndexError: 
                        result['<col>_mismatch']+=1
                        metric_logger.info('Index Error detected in split by <CEL>, counted as <col>_mismatch_error')
                        metric_logger.info(f'result: {result}')
                        continue
                    cel_ref=cols_ref[i].split(' <CEL> ')

                else: #if cell has only 1 element
                    metric_logger.info('This cell has 1 value')
                    
                    # sets cel_pred/cel_ref as a list of only 1 element of cols_pred[i]/cols_ref[i]
                    # so now, cel_pred/cel_ref will only have len of 1, which will then be able to conduct the same cel_match calc as with cells with mult values
                    cel_pred=[cols_pred[i]]
                    cel_ref=[cols_ref[i]]
                #call cel_match helper function
                result=cel_match(mode=mode,curr_header=headers[i],cel_pred=cel_pred,cel_ref=cel_ref,result=result)
                metric_logger.info(f'result: {result}')
                

        #create final dicaiontry to be returned
        final={}
        for key,val in result.items(): 
            #if it's single value: (errors)
            if isinstance(val, int):
                final[key]=val
            else: #that should be dictionary
                assert(isinstance(val,dict))
                #if ele_total=0, make it 1 to prevent error
                if val['ele_match']==0 and val['ele_total']==0: val['ele_total']=1
                final[key]=val['ele_match']/val['ele_total']*100
        metric_logger.info(f'final: {final}')
        metric_logger.info('\n---------End of evaluation epoch---------')
        return final