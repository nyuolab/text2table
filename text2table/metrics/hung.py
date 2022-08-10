import numpy as np
from scipy.optimize import linear_sum_assignment

# below code is a based on Gavin's code for icd_code matching
# func is the custom distance function
def assign(preds, refs,func):
    cost_matrix = np.array([[func(pred, label) for pred in preds] for label in refs])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind,col_ind