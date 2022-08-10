from text2table.metrics.hung import assign
import numpy as np

import evaluate

# for cel-wise prediction/reference
class ExactMatch(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo()

    # the "distance" function: calculates number of wrong chars
    # to be passed to assign() from hung.py
    def get_wrong_char(ele_pred,ele_ref):
        char_right=0 # counts number of chars matching
        char_len=len(ele_ref) # counts number of charcters in this column cell
        for c,d in zip(ele_pred,ele_ref): #c and d are each char in word a,b
            if c==d: char_right+=1 #if c,d match, count as char_right
        #find the length of the longer element/word
        max_len=max(len(ele_pred),len(ele_ref))
        #find char_wrong
        char_wrong=max_len-char_right
        return char_wrong
    
    # main function here
    def _compute(
        self,
        mode,cel_pred,cel_ref
    ):
        row_ind,col_ind=assign(cel_pred,cel_ref,get_wrong_char)
        return cel_pred[row_ind],cel_ref[col_ind]

       
        
