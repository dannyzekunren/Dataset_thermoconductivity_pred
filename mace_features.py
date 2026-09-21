import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymatgen.io.ase import AseAtomsAdaptor
bridge = AseAtomsAdaptor()
def kappa_wlmae(y_true, y_pred, p=2, log_base='e', eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # weights: full weight <=2; decay as (2/y)^p above 2
    w = np.minimum(1.0, (2.0 / np.maximum(y_true, eps))**p)
    # log error (natural or base-10)
    if log_base == 'e':
        err = np.abs(np.log(np.maximum(y_pred, eps)) - np.log(np.maximum(y_true, eps)))
    elif log_base == '10':
        err = np.abs(np.log10(np.maximum(y_pred, eps)) - np.log10(np.maximum(y_true, eps)))
    else:
        raise ValueError("log_base must be 'e' or '10'")
    return np.sum(w * err) / np.sum(w)

def symm_structure(structure):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    # Save the symm structures
    sga = SpacegroupAnalyzer(structure, symprec=1e-3)
    refined_struc = sga.get_refined_structure()
    sga = SpacegroupAnalyzer(refined_struc,symprec=0.01)
    strc_symmetry = sga.get_symmetrized_structure()
    return strc_symmetry

#split = 'random_split'
#split = 'space_group_split'
split = 'ood_split'
# Load any split
with open(f'processed_splits/{split}.pkl', 'rb') as f:
    data = pickle.load(f)
# Access features
X_train = data['X_train']
X_test = data['X_test']
structures = X_train['symmetrized_structures']  # Pymatgen Structure objects
ase_structures = [bridge.get_atoms(s) for s in structures]
structures_test = X_test['symmetrized_structures']
ase_structures_test = [bridge.get_atoms(s) for s in structures_test]

wyckoff = X_train['wyckoff_letters']
space_groups = X_train['spacegroup_numbers']
wyckoff_test = X_test['wyckoff_letters']
space_groups_test = X_test['spacegroup_numbers']
band_gaps = X_train['band_gap']  # None if no API key
band_gaps_test = X_test['band_gap']
df_X_train = pd.DataFrame(X_train)
df_X_test = pd.DataFrame(X_test)

# Access targets
y_train = data['y_train_log_klat']  # Log-transformed
y_test = data['y_test_log_klat']
y_train_original = data['y_train_klat']  # Original values
y_test_original = data['y_test_klat']
df_y_train = pd.DataFrame(y_train)
df_y_test = pd.DataFrame(y_test)


from mace.calculators import MACECalculator,mace_mp
mace_calculator = mace_mp(model = 'medium-mpa-0')

mace_descriptors = [mace_calculator.get_descriptors(ase_struc) for ase_struc in ase_structures]
mace_descriptors_test = [mace_calculator.get_descriptors(ase_struc) for ase_struc in ase_structures_test]

# save the descriptors
root_dir = 'mace_mpa/'
with open(root_dir + split +'_mace_descriptors_train.pkl', 'wb') as f:
    pickle.dump(mace_descriptors, f)

with open(root_dir + split +'_mace_descriptors_test.pkl', 'wb') as f:
    pickle.dump(mace_descriptors_test, f)