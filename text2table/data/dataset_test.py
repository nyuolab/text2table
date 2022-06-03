import datasets

mimic = datasets.load_dataset("./dataset_loading_script.py", split = "train")
print(mimic)
print(mimic[0])