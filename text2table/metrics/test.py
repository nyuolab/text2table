
a={}
a['s']=0

# import numpy as np
# import pandas as pd
# from sklearn import metrics

# from sklearn.metrics import classification_report

# ref=pd.Series(['1,4,5','2,4','3,1'])
# #ref=pd.Series(['1,4,5','2,4','3,1'])
# print(ref)
# print()
# ref_sep=ref.str.get_dummies(sep=',')
# print("ref_sep: ",ref_sep)
# print()

# pred=pd.Series(['4,5','1,4',"7,8"])
# pred_sep=pred.str.get_dummies(sep=',')
# print(pred_sep)
# print()
# pred_sep=pred_sep.reindex(columns=ref_sep.columns).fillna(0)
# print("pred_sep: ",pred_sep)
# print()

# f1=metrics.f1_score(pred_sep, ref_sep, average="micro")
# print("f1: ",f1)
# print()

# exit(0)

# y_true = [[1,2,3],[1,2],
# [2,3,4]]

# y_pred = [[2,3],[1,2],[4,5,3]]
# print(classification_report(y_true, y_pred))

# exit(0)
# if 1==1:
#     def t():
#         return 1
#     c=t
#     print(c)


# exit(0)
# y_true = [[1,2,3],[1,2],
# [2,3,4]]

# y_pred = [[2,3],[1,2],[4,5,3]]
# print(classification_report(y_true, y_pred))

# exit(0)


# ref=pd.Series(['1,4,5','2,4','3,1'])
# print(ref)
# print()
# ref_sep=ref.str.get_dummies(sep=',')
# print("ref_sep: ",ref_sep)
# print()

# pred=pd.Series(['4,5','1,4',"7,8"])
# pred_sep=pred.str.get_dummies(sep=',')
# print(pred_sep)
# print()
# pred_sep=pred_sep.reindex(columns=ref_sep.columns).fillna(0)
# print("pred_sep: ",pred_sep)
# print()

# f1=metrics.f1_score(pred_sep, ref_sep, average="micro")
# print("f1: ",f1)
# print()

# exit(0)
# # a='132'
# # b='234'
# # c='524'

# # for x,y,z in zip(a,b,c):
# #     print(x)
# #     print(y)
# #     print(z)
# #     print()

# # a=['DIAG_ICD9', 'PROC_ICD9', 'PRESCRIPTION', 'DRG_CODE','LAB_MEASUREMENT','HOSPITAL_EXPIRE_FLAG', 'GENDER', 'CPT_CD','DOB']

# # b=[f'<{x}>' for x in a]

# # print(b)

# print("asdf"==np.nan)

# exit(0)
# from scipy.optimize import linear_sum_assignment
# import numpy as np

# def wrong_char(ele_pred,ele_ref):
#     char_right=0 # counts number of chars matching
#     char_len=len(ele_ref) # counts number of charcters in this column cell
#     for c,d in zip(ele_pred,ele_ref): #c and d are each char in word a,b
#         if c==d: char_right+=1 #if c,d match, count as char_right
#     #find the length of the longer element/word
#     max_len=max(len(ele_pred),len(ele_ref))
#     #find char_wrong
#     char_wrong=max_len-char_right
#     return char_wrong

# refs=np.array(['a1','b2','c3'])
# preds=np.array(['d7','b1'])

# cost_matrix = np.array([[wrong_char(pred, label) for pred in preds] for label in refs])
# print(cost_matrix)

# # if cost_matrix.shape[1]<cost_matrix.shape[0]:
# #     cost_matrix=np.append(cost_matrix,np.zeros(cost_matrix.shape[0]).reshape(-1,1),axis=1)
# # print(cost_matrix)

# row_ind, col_ind = linear_sum_assignment(cost_matrix)
# pred_assigned = preds[row_ind]
# label_assigned = refs[col_ind]

# print(pred_assigned)
# print(label_assigned)

# # import numpy as np
# # from scipy.optimize import linear_sum_assignment

# # matrix = np.array([
# #     [10.01,     10.02,  8.03,       11.04],
# #     [9.05,      8.06,   500.07,     1.08],
# #     [9.09,      7.11,   4.11,       1000.12]
# # ])
# # print(matrix.shape)
# # exit(0)
# # row_ind, col_ind = linear_sum_assignment(matrix) 
# # #print('\nSolution:', matrix[row_ind, col_ind].sum())

# # print(row_ind)
# # print(col_ind)

# # pred_assigned = matrix[row_ind]
# # label_assigned = matrix[:,col_ind]

# # print(pred_assigned)
# # print()
# # print(label_assigned)