import pickle
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm


def structure2poscar(source_file_path, target_file_path):
    with open(source_file_path, 'rb') as f:
        data = pickle.load(f)
    X_train_struct = data['X_train']['structures']
    X_train_id = np.array(data['X_train']['mp_ids'])
    X_test_struct = data['X_test']['structures']
    X_test_id = np.array(data['X_test']['mp_ids'])
    success_num=0
    for i, struct in tqdm(enumerate(X_train_struct)):
        struct.to(fmt="poscar", filename=os.path.join(target_file_path,f'{X_train_id[i]}'))
        success_num+=1
    print(f'Train set: {success_num} structures converted to POSCAR format.')
    success_num=0
    for i, struct in enumerate(X_test_struct):
        struct.to(fmt="poscar", filename=os.path.join(target_file_path,f'{X_test_id[i]}'))
        success_num+=1
    print(f'Test set: {success_num} structures converted to POSCAR format.')

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--source_file_path', type=str, required=True, help='Path to the source pickle file')
    arg_parser.add_argument('--target_file_path', type=str, required=True, help='Path to save the output POSCAR files')
    args = arg_parser.parse_args()

    structure2poscar(args.source_file_path, args.target_file_path)