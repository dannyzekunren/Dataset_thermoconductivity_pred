#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-xx-xx
Description: [Brief description of the script's purpose]
"""
import os
import json
import numpy as np
from fractions import Fraction
from sklearn.preprocessing import LabelEncoder

# De Graef, M., & McHenry, M. E. (2012). Structure of materials: an introduction to crystallography, diffraction and symmetry (Second edition, fully revised and updated.). Cambridge University Press.
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

base_dir = os.path.dirname(__file__)
strings = load_json(os.path.join(base_dir, "strings.json"))              # space group number -> generator string
mapping = load_json(os.path.join(base_dir, "char_to_vec.json"))          # Characters -> vector/matrix entries

def string_to_matrix(gen_str, mapping):
    has_inversion = gen_str[0] == "1"
    n_generators = int(gen_str[1])
    pointer = 2

    generators = []

    if has_inversion:
        M_inv = np.eye(4, dtype=float)
        M_inv[:3, :3] = -np.eye(3, dtype=float)  # R = -I
        M_inv[:3, 3] = 0                            # t = 0
        generators.append(M_inv)
    else:
        M = np.zeros((4, 4), dtype=float)
        generators.append(M)

    for _ in range(n_generators):
        symbol = gen_str[pointer]
        trans_keys = gen_str[pointer+1:pointer+4]
        pointer += 4

        # Rotation matrix
        R = np.array(mapping[symbol], dtype=float)

        # Translation vector
        t = np.zeros(3, dtype=float)
        for i, key in enumerate(trans_keys):
            t += np.array(Fraction(mapping[key]), dtype=float)

        # 4×4 affine matrix
        M = np.eye(4, dtype=float)
        M[:3,:3] = R
        M[:3,3] = t
        generators.append(M)

    # Origin shift
    if pointer < len(gen_str) and gen_str[pointer] == "1":
        origin_keys = gen_str[pointer + 1:pointer + 4]
        t_origin = np.zeros(3, dtype=float)
        for i, key in enumerate(origin_keys):
            t_origin += np.array(Fraction(mapping[key]), dtype=float)
        origin_shift = t_origin
    else:
        origin_shift = np.zeros(3, dtype=float)

    return generators, origin_shift

def fit_label_encoder():
    """
    Fit a LabelEncoder over all characters that appear in the generator strings.
    """
    all_chars = sorted(set("".join(strings.values())))
    le = LabelEncoder()
    le.fit(list(all_chars))
    print("Fitted LabelEncoder with classes:", le.classes_)
    return le

def string_to_encoded_array(gen_str, le, max_len=None):
    """
    Encode a generator string into a fixed-length numeric array using LabelEncoder.

    Args:
        gen_str (str): e.g. "103Axyz..." (generator string)
        le (LabelEncoder): fitted encoder over all possible symbols
        max_len (int): maximum length of the output array; if None, use the length of the longest string
    Returns:
        np.ndarray: encoded numeric feature (shape = (max_len,))
    """
    if max_len is None:
        max_len = max(len(s) for s in strings.values())

    encoded = le.transform(list(gen_str))
    if len(encoded) < max_len:
        padded = np.pad(encoded, (0, max_len - len(encoded)), 'constant', constant_values=-1)
    else:
        padded = encoded[:max_len]
    return padded

def get_space_group_gen_matrix(sg_number: None | int = None):
    if sg_number is None:
        all_gens = {}
        for sg, gen_str in strings.items():
            generators, origin_shift = string_to_matrix(gen_str, mapping)
            all_gens[int(sg)] = (generators, origin_shift)
        return all_gens
    else:
        sg_number = str(sg_number)
        if sg_number not in strings:
            raise ValueError(f"Please provide a valid space group number (1-230). Got: {sg_number}")
        gen_str = strings[sg_number]
        generators, origin_shift = string_to_matrix(gen_str, mapping)
        return generators, origin_shift

def get_space_group_gen_label(le, sg_number: None | int = None):
    if sg_number is None:
        all_labels = {}
        for sg, gen_str in strings.items():
            gen_label = string_to_encoded_array(gen_str, le)
            all_labels[int(sg)] = gen_label
        return all_labels
    else:
        sg_number = str(sg_number)
        if sg_number not in strings:
            raise ValueError(f"Please provide a valid space group number (1-230). Got: {sg_number}")
        gen_str = strings[sg_number]
        gen_label = string_to_encoded_array(gen_str, le)
        return gen_label


if __name__ == "__main__":
    # Single space group
    sg_num = 225
    gens, origin = get_space_group_gen_matrix(sg_num)
    print(f"Space group {sg_num} generators: {len(gens)}")
    print("Origin shift:", origin)

    # All space groups
    all_gens = get_space_group_gen_matrix()
    print(f"Total space groups loaded: {len(all_gens)}")
    print(f"Generators for space group 1: {len(all_gens[1][0])}")
