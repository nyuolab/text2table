import pandas as pd
import numpy as np

# read in the dataset
dataset = pd.read_csv("/gpfs/data/oermannlab/project_data/text2table/complete_v2/dataset_v2.csv")

# empirical entropy
def emp_entropy(series):
    # dummify the column on which we are calculating empirical entropy
    dummified = series.astype(str).dropna().str.get_dummies(sep=" <CEL> ")
    # count the number of samples for every unique combinations of labels
    unique_combs = dummified.groupby(list(dummified.columns)).size().reset_index().rename(columns={0: "count"})
    count = unique_combs["count"].astype(float)
    # calculate the entropy
    entropy = (count/count.sum())*np.log2(count/count.sum())
    return entropy.sum() * -1

# calculate entropies for every column of interest
dict = {}
for x in [x for x in list(dataset.columns) if x not in ["DOB", "HADM_ID", "TEXT"]]:
    dict[x] = emp_entropy(dataset[x])

# display and save the results
entropy_df = pd.DataFrame(dict, index=["empirical entropy"])
print(entropy_df)
entropy_df.to_csv("imbalanceness.csv")