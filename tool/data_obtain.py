import pickle
import pandas as pd
import numpy as np
import os
import argparse

def data_obtain(source_file_path, target_file_path,target_name):
    """
    Load data from a pickle file and convert it to a pandas DataFrame.

    Parameters:
    file_path (str): The path to the pickle file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    with open(source_file_path, 'rb') as f:
        data = pickle.load(f)

    X_train = data['X_train']
    X_train_id = np.array(X_train['mp_ids'])
    y_train_klat = data['y_train_klat']
    y_train = np.log10(y_train_klat)
    train_data = [X_train_id, y_train]
    train_data_pd = pd.DataFrame(train_data).transpose()

    X_test = data['X_test']
    X_test_id = np.array(X_test['mp_ids'])
    y_test_klat = data['y_test_klat']
    y_test = np.log10(y_test_klat)
    test_data = [np.array(X_test_id), y_test]
    test_data_pd = pd.DataFrame(test_data).transpose()


    train_data_pd.to_csv(os.path.join(target_file_path,f'{target_name}_train_data.csv'), index=False, header=False)
    print(f"Train data saved to {os.path.join(target_file_path,f'{target_name}_train_data.csv')}")
    test_data_pd.to_csv(os.path.join(target_file_path,f'{target_name}_test_data.csv'), index=False, header=False)
    print(f"Test data saved to {os.path.join(target_file_path,f'{target_name}_test_data.csv')}")

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--source_file_path', type=str, required=True, help='Path to the source pickle file')
    arg_parser.add_argument('--target_file_path', type=str, required=True, help='Path to save the output CSV files')
    arg_parser.add_argument('--target_name', type=str, required=True, help='Base name for the output CSV files')
    args = arg_parser.parse_args()
    
    data_obtain(args.source_file_path, args.target_file_path,args.target_name)