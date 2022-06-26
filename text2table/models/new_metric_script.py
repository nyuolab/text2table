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

#logging config:
#define filename: date_time_run.log
a=datetime.datetime.now()
#os.makedirs('eval_logs',exist_ok=True)
b=a.strftime("eval_logs/%m_%d_%H:%M:%S")+'_eval.log'
#config:

logging.info('---------Start of evaluation epoch---------')

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

# don't need the below function
    # def _download_and_prepare(self, dl_manager):
    #     """Optional: download external resources useful to compute the scores"""
    #     # TODO: Download external resources if needed
    #     bad_words_path = dl_manager.download_and_extract(BAD_WORDS_URL)
    #     self.bad_words = {w.strip() for w in open(bad_words_path, encoding="utf-8")}

    def _compute(self, predictions, references): #predictions, references both in a batch
        # result={}
        # #initiate dictionary
        # for col in headers:
        #     for c in self.config_name:
        #         result[f'{c}_{col}']={'ele_match':0,'ele_total':0}
        # #can't find <row> separator
        # result['<row>_error']=0
        # #unequal number of columns
        # result['<col>_mismatch']=0
        

        #--test
        logging.info('Successfulyl enters metric')
        """Returns the scores"""
        # TODO: Compute the different scores of the metric
        #replace the below with function 

        #modes
        if self.config_name == '20':
            perc=0.2
        elif self.config_name == '10':
            perc=0.1
        elif self.config_name == '0':
            perc=0
        else: raise ValueError(f"Invalid config name for ColMatch: {c}. Please use '0', '10', or '20'.")
        b=a.strftime(f"{c}_eval_logs/%m_%d_%H:%M:%S")+'_eval.log'
        logging.basicConfig(filename=b, level=logging.DEBUG,format='%(asctime)s - %(message)s',datefmt='%d-%b-%y %H:%M:%S')

        #get column header from first reference
        headers=references[0].split(' <ROW> ')[0].split(' <COL> ')
        headers[0]=headers[0].replace('<s>','')
        logging.info('headers: '+','.join(headers))

        #initiate result
        result={}
        for col in headers: result[col]={'ele_match':0,'ele_total':0}

        #--test
        count=0
        #iterate thru rows/inputs
        for row_pred,row_ref in zip(predictions, references):
            #--test
            count+=1
            #logging.info('current row in batch: ',count)

            #split pred_str by columns as a list
            #replace <pad> since annoying
            row_ref=row_ref.replace('<pad>','')
            row_pred=row_pred.replace('<pad>','')
            #--test
            #logging.info('row_ref: ',row_ref)
            #logging.info('row_pred: ',row_pred)

            #error: first evaluation may look like </s><s>, need to skip
            if ' <ROW> ' not in row_pred: 
                result['<row>_error']+=1
                #--test
                #logging.info('result: ',result)
                continue
            cols_pred=row_pred.split(' <ROW> ')[1].split(' <COL> ')
            cols_ref=row_ref.split(' <ROW> ')[1].split(' <COL> ')
            #--test
            #logging.info('cols_pred: ',cols_pred)
            #logging.info('cols_ref: ',cols_ref)

            #if length mismatch, log as error
            if len(cols_pred)!=len(cols_ref):
                result['<col>_mismatch']+=1
                #logging.info('result: ',result)
                continue

            # iterate thru columns
            for i in range(len(headers)):  
                #--test
                #logging.info('current header: ',headers[i])
                if ' <CEL> ' in cols_ref[i]: # use ref to be safe, if ref cell has multiple elements
                    #--test
                    #logging.info('cell with mult values')
                    try: #if last column doesn't exist (a consequence of mismatch length of columns)
                        cel_pred=cols_pred[i].split(' <CEL> ')
                    except IndexError: 
                        result['<col>_mismatch']+=1
                        #--test
                        #logging.info('Index Error')
                        #logging.info('result: ',result)
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
                    #--test
                    #logging.info('cell with 1 value')
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
                
                #--test
                #logging.info('result: ',result)
                

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
        #--test
        #logging.info('\final: ',final)
        #logging.info('---------End of evaluation epoch---------')
        print('\nfinal: ',final)
        return final
