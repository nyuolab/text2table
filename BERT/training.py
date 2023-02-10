import argparse
from dataset_loading import loading_dataset

if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()

    # path
    data_dir="/gpfs/data/oermannlab/project_data/text2table/complete_v2/train_test_data/"

    [train, dev, test] = loading_dataset(data_dir, args.task.split('-'))

    

    
