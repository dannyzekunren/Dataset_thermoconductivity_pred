#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-xx-xx
Description: [Brief description of the script's purpose]
"""
import pickle
import os
import numpy as np
import torch
from dscribe.descriptors import SOAP
from matminer.featurizers.composition import ElementProperty
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from generator_feature import get_space_group_gen_matrix, get_space_group_gen_label, fit_label_encoder

def detect_species(structures):
    species_set = set()
    for s in structures:
        for site in s:
            species_set.add(site.specie.symbol)
    return sorted(list(species_set))


def featurize_compositions(X_train: dict) -> np.ndarray:
    comps = [s.composition for s in X_train["structures"]]
    ep = ElementProperty.from_preset(preset_name="matminer")
    composition_features = [ep.featurize(comp) for comp in tqdm(comps, desc="Featurizing compositions")]
    return np.array(composition_features)

def get_soap_embedding(X_train: dict, X_test: dict, pca=True) -> (np.ndarray, np.ndarray):
    train_structures = X_train["symmetrized_structures"]
    test_structures = X_test["symmetrized_structures"]
    species = sorted(list(set(
        detect_species(train_structures) +
        detect_species(test_structures)
    )))
    soap = SOAP(species=species,
                r_cut=6.0,
                n_max=8, l_max=6,
                sparse=False,
                periodic=True,
                average="inner",
                compression={"mode": "mu2", "species_weighting": None}
                )

    train_envs=[]
    for s in tqdm(train_structures, desc="Featurizing train structures with SOAP"):
        ase_atoms = AseAtomsAdaptor.get_atoms(s)
        env = soap.create(ase_atoms)
        train_envs.append(env)

    test_envs=[]
    for s in tqdm(test_structures, desc="Featurizing test structures with SOAP"):
        ase_atoms = AseAtomsAdaptor.get_atoms(s)
        env = soap.create(ase_atoms)
        test_envs.append(env)

    if pca:
        pca = PCA(n_components=100)
        train_envs = pca.fit_transform(np.array(train_envs))
        test_envs = pca.transform(np.array(test_envs))

    return np.array(train_envs), np.array(test_envs)

def flatten_symmop(op):
    rot = op.rotation_matrix.flatten()
    tau = op.translation_vector.flatten()
    return np.concatenate([rot, tau])

def featurize_symmetry_by_padding(X_train, target_dim=64) -> np.ndarray:
    """
    Featurize symmetry operations by padding to a fixed size. (192 * 12)
    Args:
        X_train: dict
    Returns:
        np.ndarray
    """
    structures = X_train["symmetrized_structures"]
    features = []
    iterator = tqdm(structures, desc="Featurizing symmetry operations")
    for s in iterator:
        try:
            sga = SpacegroupAnalyzer(s)
            ops = sga.get_symmetry_operations()
            ops_flat = np.array([flatten_symmop(op) for op in ops])

            padded = np.zeros((192, 12))  # Max 192 symmetry operations
            padded[:ops_flat.shape[0], :] = ops_flat
            features.append(padded.flatten())

        except Exception as e:
            print(f"Error processing structure: {e}. Using zero vector.")
            features.append(np.zeros(192 * 12))

    features = np.array(features)
    pca = PCA(n_components=target_dim)
    features = pca.fit_transform(features)

    return features

def featurize_symmetry_by_bin_matrix(X_train, one_hot=False) -> np.ndarray:
    structures = X_train["symmetrized_structures"]
    features = []
    iterator = tqdm(structures, desc="Featurizing Space Groups")
    for s in iterator:
        try:
            sga = SpacegroupAnalyzer(s)
            spg_number = sga.get_space_group_number()
            group = Group(spg_number)
            if one_hot:
                id, bin_matrix = group.get_spg_representation()
                feat = np.hstack([[id], bin_matrix.ravel()])
            else:
                feat = group.get_spg_symmetry_object().to_matrix_representation().flatten()
            features.append(feat)

        except Exception as e:
            print(f"Error processing structure: {e}. Using zero vector.")
            features.append(np.zeros(270))  # 15 * 18

    features = np.array(features)
    return features

def featurize_symmetry_by_gen_matrix(X_train) -> np.ndarray:
    """
    Featurize symmetry operations by generators.
    Args:
        X_train: dict
    Returns:
        np.ndarray
    """
    structures = X_train["symmetrized_structures"]
    features = []
    iterator = tqdm(structures, desc="Featurizing generators")
    for s in iterator:
        try:
            sga = SpacegroupAnalyzer(s)
            spg_number = sga.get_space_group_number()
            gens, origin_shift = get_space_group_gen_matrix(spg_number)
            feat = np.zeros((1, 8*3*4 + 3))  # 8 gens * 3 * 4 + 3 origin shift
            gen_flat = np.concatenate([g[:3, :4].flatten() for g in gens])
            feat[0, :len(gen_flat)] = gen_flat
            feat[0, 8*12:] = np.array(origin_shift)
            features.append(feat.flatten())

        except Exception as e:
            print(f"Error processing structure: {e}. Using zero vector.")
            features.append(np.zeros(99))  # 8 * 3 * 4 + 3

    features = np.array(features)
    return features

def featurize_symmetry_by_gen_label(X_train) -> np.ndarray:
    """
    Featurize symmetry operations by generators.
    Args:
        X_train: dict
    Returns:
        np.ndarray
    """
    structures = X_train["symmetrized_structures"]
    features = []
    iterator = tqdm(structures, desc="Featurizing generators")
    le = fit_label_encoder()
    for s in iterator:
        try:
            sga = SpacegroupAnalyzer(s)
            spg_number = sga.get_space_group_number()
            feat = get_space_group_gen_label(le, spg_number)
            features.append(feat.flatten())

        except Exception as e:
            print(f"Error processing structure: {e}. Using zero vector.")
            features.append(np.zeros(34))

    features = np.array(features)
    return features

def featurize_symmetry_by_statistics(X_train, normalize=True) -> np.ndarray:
    """
    Featurize symmetry operations by statistics.
    Args:
        X_train: dict
        normalize: bool, whether to normalize the features
    Returns:
        np.ndarray
    """
    structures = X_train["symmetrized_structures"]
    features = []
    iterator = tqdm(structures, desc="Featurizing symmetry operations")
    for s in iterator:
        try:
            sga = SpacegroupAnalyzer(s)
            ops = sga.get_symmetry_operations()
            n_ops = len(ops)
            ops_flat = np.array([flatten_symmop(op) for op in ops])

            mean = np.mean(ops_flat, axis=0)
            std = np.std(ops_flat, axis=0)
            feat = np.concatenate([[n_ops / 192], mean, std])
            features.append(feat)

        except Exception as e:
            print(f"Error processing structure: {e}. Using zero vector.")
            features.append(np.zeros(49))  # 1 + 2 * 12

    features = np.array(features)
    if normalize:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1.0
        features = (features - mean) / std

    return features

def clean_nan(X: np.ndarray, max_row_nan=100):
    n_samples, n_features = X.shape

    nan_counts_col = np.isnan(X).sum(axis=0)
    col_mask = nan_counts_col <= max_row_nan
    X_col_clean = X[:, col_mask]
    print(f"Removed {n_features - X_col_clean.shape[1]} features with more than {max_row_nan} NaNs.")

    nan_counts_row = np.isnan(X_col_clean).sum(axis=1)
    row_mask = nan_counts_row == 0
    X_clean = X_col_clean[row_mask]
    print(f"Removed {n_samples - X_clean.shape[0]} samples with any NaNs.")
    print(f"Original shape: {X.shape}, Cleaned shape: {X_clean.shape}")
    return X_clean, row_mask


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        print("No GPU available, using CPU. This may be slow.")
    else:
        print(f"Using device: {device}")

    base_dir = os.path.dirname(__file__)
    with open("/mnt/data/home/jhwang/projects/kan/dataset/processed_splits/ood_split.pkl", "rb") as f:
        data_splits = pickle.load(f)

    print(data_splits.keys())

    X_train, y_train = data_splits["X_train"], data_splits["y_train_log_klat"].reshape(-1, 1)
    X_test, y_test = data_splits["X_test"], data_splits["y_test_log_klat"].reshape(-1, 1)
    print(X_train.keys())

    # Feature selection
    X_train_composition = featurize_compositions(X_train)
    X_test_composition = featurize_compositions(X_test)
    print("Composition feature shape:", X_test_composition.shape)

    X_train_soap, X_test_soap = get_soap_embedding(X_train, X_test, pca=False)
    print("SOAP feature shape:", X_test_soap.shape)

    X_train_symmetry = featurize_symmetry_by_bin_matrix(X_train)
    X_test_symmetry = featurize_symmetry_by_bin_matrix(X_test)
    print("Symmetry feature shape:", X_test_symmetry.shape)

    # X_train_numerical = np.column_stack([
    #     X_train["e_above_hull"],
    #     X_train["formation_energy_per_atom"],
    #     X_train["band_gap"],
    #     X_train["total_magnetization"],
    # ])
    #
    # X_test_numerical = np.column_stack([
    #     X_test["e_above_hull"],
    #     X_test["formation_energy_per_atom"],
    #     X_test["band_gap"],
    #     X_test["total_magnetization"],
    # ])

    X_train_full = np.hstack([
        X_train_composition,
        X_train_soap,
        X_train_symmetry,
        # X_train_numerical
    ])

    X_test_full = np.hstack([
        X_test_composition,
        X_test_soap,
        X_test_symmetry,
        # X_test_numerical
    ])

    X_train_cleaned, row_mask_train = clean_nan(X_train_full)
    X_test_cleaned, row_mask_test = clean_nan(X_test_full)

    y_train_cleaned = y_train[row_mask_train]
    y_test_cleaned = y_test[row_mask_test]

    # Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_cleaned)
    X_test_scaled = scaler.transform(X_test_cleaned)


    dataset = {
        'train_input': torch.tensor(X_train_scaled, dtype=torch.float32).to(device),
        'train_label': torch.tensor(y_train_cleaned, dtype=torch.float32).to(device),
        'test_input': torch.tensor(X_test_scaled, dtype=torch.float32).to(device),
        'test_label': torch.tensor(y_test_cleaned, dtype=torch.float32).to(device)
    }

    with open("/mnt/data/home/jhwang/projects/kan/dataset/featurized_data/ood_scaled_full_soap_bin_matrix_sym_logk.pkl", "wb") as f:
        pickle.dump(dataset, f)
