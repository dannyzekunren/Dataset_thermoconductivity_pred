#!/usr/bin/env python

"""Module to train for a folder with formatted dataset."""
import csv
import os
import sys
import ast
from jarvis.core.atoms import Atoms
from aliegnn.data_split import get_train_val_loaders
from aliegnn.train import train_dgl
from aliegnn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
import os

# Set PyTorch memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import argparse
import torch
import random
import time
import gc

parser = argparse.ArgumentParser(
    description="Atomistic Line Graph Neural Network"
)
parser.add_argument(
    "--root_dir",
    default="./",
    help="Folder with id_props.csv, structure files",
)
parser.add_argument(
    "--config_name",
    default="config_example.json",
    help="Name of the config file",
)

parser.add_argument(
    "--id_prop_file_train",
    default="random_train_data.csv",
    help="Name of the id_prop file, default id_prop.csv",
)
parser.add_argument(
    "--id_prop_file_test",
    default="random_test_data.csv",
    help="Name of the id_prop file, default id_prop.csv",
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)

parser.add_argument(
    "--keep_data_order",
    default=False,
    help="Whether to randomly shuffle samples, True/False",
)

parser.add_argument(
    "--classification_threshold",
    default=None,
    help="Floating point threshold for converting into 0/1 class"
    + ", use only for classification tasks",
)

parser.add_argument(
    "--batch_size", default=None, help="Batch size, generally 64"
)

parser.add_argument(
    "--epochs", default=None, help="Number of epochs, generally 300"
)

parser.add_argument(
    "--output_dir",
    default="./",
    help="Folder to save outputs",
)

parser.add_argument(
    "--cv",
    type=int,
    default=0,
    help="Use cross-validation",
)

parser.add_argument(
    "--sample_size",
    type=int,
    default=None,
    help="Sample size, if None, use all data",
)

def train_for_folder(
    root_dir="examples/sample_data",
    config_name="config.json",
    keep_data_order=False,
    classification_threshold=None,
    batch_size=None,
    epochs=None,
    file_format="poscar",
    output_dir=None,
    cv=0,
):
    """Train for a folder."""
    # config_dat=os.path.join(root_dir,config_name)
    id_prop_dat_train = os.path.join(root_dir, args.id_prop_file_train)
    id_prop_dat_test = os.path.join(root_dir, args.id_prop_file_test)
    if type(config_name) is str:
        config_path = os.path.join(root_dir, config_name)
        config = loadjson(config_path)
    else:
        config = config_name
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)

    config.keep_data_order = keep_data_order
    if classification_threshold is not None:
        config.classification_threshold = float(classification_threshold)
    if output_dir is not None:
        config.output_dir = output_dir
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        config.epochs = int(epochs)
    with open(id_prop_dat_train, "r") as f:
        reader = csv.reader(f)
        data_train = [row for row in reader]
    with open(id_prop_dat_test, "r") as f:
        reader = csv.reader(f)
        data_test = [row for row in reader]

    if args.sample_size is not None:
        if args.sample_size > len(data_train):
            raise ValueError(
                f"Sample size {args.sample_size} is greater than the number of samples {len(data_train)}."
            )
        data_train = random.sample(data_train, args.sample_size)
    else:
        args.sample_size = len(data_train)
    print (f"Using {args.sample_size} samples for training")


    dataset_train = []
    dataset_test = []
    n_outputs = []
    multioutput = False
    lists_length_equal = True
    for i in data_train:
        info = {}
        file_name = i[0]
        file_path = os.path.join(root_dir, file_name)
        if file_format == "poscar":
            atoms = Atoms.from_poscar(file_path)
        elif file_format == "cif":
            atoms = Atoms.from_cif(file_path)
        elif file_format == "xyz":
            # Note using 500 angstrom as box size
            atoms = Atoms.from_xyz(file_path, box_size=500)
        elif file_format == "pdb":
            # Note using 500 angstrom as box size
            # Recommended install pytraj
            # conda install -c ambermd pytraj
            atoms = Atoms.from_pdb(file_path, max_lat=500)
        else:
            raise NotImplementedError(
                "File format not implemented", file_format
            )

        info["atoms"] = atoms.to_dict()
        info["jid"] = file_name

        try:
            tmp = [float(j) for j in i[1:]]  # float(i[1])
        except:
            for j in i[1:]:
               tmp = ast.literal_eval(j)
        if len(tmp) == 1:
            tmp = tmp[0]
        else:
            multioutput = True
        info["target"] = tmp  # float(i[1])
        n_outputs.append(info["target"])
        dataset_train.append(info)
    for i in data_test:
        info = {}
        file_name = i[0]
        file_path = os.path.join(root_dir, file_name)
        if file_format == "poscar":
            atoms = Atoms.from_poscar(file_path)
        elif file_format == "cif":
            atoms = Atoms.from_cif(file_path)
        elif file_format == "xyz":
            # Note using 500 angstrom as box size
            atoms = Atoms.from_xyz(file_path, box_size=500)
        elif file_format == "pdb":
            # Note using 500 angstrom as box size
            # Recommended install pytraj
            # conda install -c ambermd pytraj
            atoms = Atoms.from_pdb(file_path, max_lat=500)
        else:
            raise NotImplementedError(
                "File format not implemented", file_format
            )

        info["atoms"] = atoms.to_dict()
        info["jid"] = file_name

        try:
            tmp = [float(j) for j in i[1:]]  # float(i[1])
        except:
            for j in i[1:]:
               tmp = ast.literal_eval(j)
        if len(tmp) == 1:
            tmp = tmp[0]
        else:
            multioutput = True
        info["target"] = tmp  # float(i[1])
        n_outputs.append(info["target"])
        dataset_test.append(info)
    if multioutput:
        lists_length_equal = False not in [
            len(i) == len(n_outputs[0]) for i in n_outputs
        ]

    # print ('n_outputs',n_outputs[0])
    if multioutput and classification_threshold is not None:
        raise ValueError("Classification for multi-output not implemented.")
    if multioutput and lists_length_equal:
        config.model.output_features = len(n_outputs[0])
    else:
        # TODO: Pad with NaN
        if not lists_length_equal:
            raise ValueError("Make sure the outputs are of same size.")
        else:
            config.model.output_features = 1

    best_val_metric = float("inf")
    (
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ) = get_train_val_loaders(
        dataset_array=[],
        train_data=dataset_train,
        test_data=dataset_test,
        target=config.target,
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        standardize=config.atom_features != "cgcnn",
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=config.save_dataloader,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        max_neighbors=config.max_neighbors,
        output_features=config.model.output_features,
        classification_threshold=config.classification_threshold,
        target_multiplication_factor=config.target_multiplication_factor,
        standard_scalar_and_pca=config.standard_scalar_and_pca,
        keep_data_order=config.keep_data_order,
        output_dir=config.output_dir,
    )
    start_time = time.time()
    train_dgl(
        config,
        train_val_test_loaders=[
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ],
        cv=0,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    val_metrics_path = os.path.join(
        config.output_dir, "prediction_metrics_results_cv.csv"
    )
    if os.path.exists(val_metrics_path):
        with open(val_metrics_path, "r") as f:
            reader = csv.reader(f)
            try:
                # header = next(reader)
                val_metrics = {row[0]: float(row[1]) for row in reader}
            except (StopIteration, IndexError, ValueError) as e:
                print(f"Error reading MAE from {val_metrics_path}: {e}. Setting MAE to infinity.")
                val_metrics = {'fold':float('inf')}
        # Calculate average of all metrics (consistent with cv > 0 case)
        best_val_metric = sum(val_metrics.values()) / len(val_metrics)
        print(f"Best validation metric: {best_val_metric}")
    else:
        print(f"Validation metrics file {val_metrics_path} not found. Setting MAE to infinity.")
        best_val_metric = float('inf')

    # Save the time taken for training
    time_file_path = os.path.join(config.output_dir, "training_time.txt")
    with open(time_file_path, "w") as time_file:
        time_file.write(f"Training completed in {elapsed_time:.2f} seconds.\n")
    return best_val_metric


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        print("GPU cache cleared before training")
    
    try:
        train_for_folder(
            root_dir=args.root_dir,
            config_name=args.config_name,
            keep_data_order=args.keep_data_order,
            classification_threshold=args.classification_threshold,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            file_format=args.file_format,
            cv=args.cv,
        )
    finally:
        # Clean up GPU memory after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("GPU cache cleared after training")
