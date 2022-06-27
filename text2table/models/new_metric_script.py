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
import logging
import datetime
import os



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

# TODO: Define external resources urls if needed
BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"


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

    def _compute(self, predictions, references): #predictions, references both in a batch
        """Returns the scores"""
        # configs: matters for compute and logging
        if self.config_name == '20':
            perc=0.2
            logging.info('\n---------Start of evaluation epoch: perc 20---------')
        elif self.config_name == '10':
            perc=0.1
            logging.info('\n---------Start of evaluation epoch: perc 10---------')
        elif self.config_name == '0':
            perc=0
            logging.info('\n---------Start of evaluation epoch: perc 0---------')
        else: raise ValueError(f"Invalid config name for ColMatch: {self.config_name}. Please use '0', '10', or '20'.")


        #get column header from first reference
        headers=references[0].split(' <ROW> ')[0].split(' <COL> ')
        headers[0]=headers[0].replace('<s>','')
        logging.info('headers: '+','.join(headers))

        #initiate result dic
        result={}
        for col in headers: result[col]={'ele_match':0,'ele_total':0}
        result['<col>_mismatch']=0
        result['<row>_error']=0

        #--test var
        count=0
        #iterate thru rows/inputs
        for row_pred,row_ref in zip(predictions, references):
            #--test var
            count+=1
            logging.info(f'\ncurrent row in batch: {count}')
            #split pred_str by columns as a list
            row_ref=row_ref.replace('<pad>','')
            #replace <pad> since annoying
            row_pred=row_pred.replace('<pad>','')
            logging.info(f'row_ref: {row_ref}')
            logging.info(f'row_pred: {row_pred}')

            #error: first evaluation may look like </s><s>, need to skip
            if ' <ROW> ' not in row_pred: 
                result['<row>_error']+=1
                logging.info('<row>_error detected')
                logging.info(f'result: {result}')
                continue
            cols_pred=row_pred.split(' <ROW> ')[1].split(' <COL> ')
            cols_ref=row_ref.split(' <ROW> ')[1].split(' <COL> ')
            logging.info(f"cols_pred: {', '.join(cols_pred)}")
            logging.info(f"cols_ref: {', '.join(cols_ref)}")

            #if length mismatch, log as error
            if len(cols_pred)!=len(cols_ref):
                result['<col>_mismatch']+=1
                logging.info('<col>_mismatch detected')
                logging.info(f'result: {result}')
                continue

            # iterate thru columns
            for i in range(len(headers)):  
                logging.info(f'current header: {headers[i]}')
                if ' <CEL> ' in cols_ref[i]: # use ref to be safe, if ref cell has multiple elements
                    logging.info('This cell has multi values')
                    try: #if last column doesn't exist (a consequence of mismatch length of columns)
                        cel_pred=cols_pred[i].split(' <CEL> ')
                    except IndexError: 
                        result['<col>_mismatch']+=1
                        logging.info('Index Error detected in split by <CEL>, counted as <col>_mismatch_error')
                        logging.info(f'result: {result}')
                        continue
                    cel_ref=cols_ref[i].split(' <CEL> ')

                    #number of elements in reference
                    ele_total=len(cel_ref)
                    ele_match=0
                    #iterate thru each element in a cell
                    for a, b in zip(cel_pred, cel_ref):
                        char_wrong=0 # counts number of chars matching
                        char_len=len(b) # counts number of charcters in this column cell
                        for c,d in zip(a,b): #c and d are each char in word a,b
                            if c!=d: char_wrong+=1 #if c,d not match, count as  
                        #if number of matching chars smaller than length of word
                        if char_wrong/char_len<=perc: ele_match+=1
                else: #if cell has only 1 element
                    logging.info('This cell with 1 value')
                    #number of elements in reference
                    #number of elements in reference
                    ele_total=1
                    ele_match=0

                    char_wrong=0 # counts number of chars matching
                    char_len=len(cols_ref[i]) # counts number of charcters in this column cell
                    for c,d in zip(cols_pred[i],cols_ref[i]): #c and d are each char in word a,b
                        if c!=d: char_wrong+=1 #if c,d not match, count as  
                    #if number of matching chars smaller than length of word
                    if char_wrong/char_len<=perc: ele_match=1
                    
                #append tmp to result dic for this column
                result[headers[i]]['ele_match']+=ele_match
                result[headers[i]]['ele_total']+=ele_total
                #--test
                #logging.info('ele_match: ',ele_match)
                #logging.info('ele_total: ',ele_total)
                
                logging.info(f'result: {result}')
        return result
