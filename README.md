# Thermal Conductivity Dataset and Surrogate Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22894536.svg)](https://doi.org/10.5281/zenodo.22894536)

This repository provides curated train/test splits for lattice thermal conductivity prediction, constructed from Materials Project and [Phonix](https://huggingface.co/phonix-db) data, together with resources for training and evaluating surrogate models for thermal conductivity prediction. 

## Overview

Three splits are provided:

1. **Random 80/20 Split**: Standard baseline for comparison
2. **Space Group Disjoint Split**: Tests generalization to unseen crystal structures
3. **Out-of-Distribution Split**: Tests performance on low thermal conductivity materials

## Dataset

- **Total Samples**: 6,966 (filtered from 6,967 original entries)
- **Filtering**: Removed samples where `klat > 10^5`, `klat <= 0`, or non-finite values
- **Target**: `log(klat)` for all splits

## Data Structure

Each split pickle file contains:

### Features (X_train, X_test)
- **Structures**: Original and symmetrized pymatgen Structure objects
- **Crystallography**: Wyckoff positions, space group symbols/numbers
- **Thermal**: Crystal (kc) and phonon (kp) thermal conductivity components
- **Properties**: Formation energy, band gap, magnetization, elasticity (if MP API key provided)

### Targets (Y)
- **y_train_log_klat**, **y_test_log_klat**: Log-transformed thermal conductivity
- **y_train_klat**, **y_test_klat**: Original thermal conductivity values

## Split Details

### 1. Random 80/20 Split (Baseline)
- **Train**: 5,563 samples (79.9%)
- **Test**: 1,403 samples (20.1%)
- **Purpose**: Standard baseline for comparison

### 2. Space Group Disjoint Split
- **Train**: 5,573 samples (80.0%), 139 space groups
- **Test**: 1,393 samples (20.0%), 31 space groups
- **Purpose**: Tests generalization to unseen crystal structures
- **Key**: Zero space group overlap between train and test

### 3. Out-of-Distribution Split
- **Train**: 5,042 samples (72.4%), klat > 0.8
- **Test**: 1,914 samples (27.5%), klat < 1.0
- **Purpose**: Tests performance on low thermal conductivity materials
- **Key**: Clean separation with gap region (0.8 < klat < 1.0)

## Installation

```bash
pip install -r requirements.txt
```

### Optional: Materials Project API

To fetch additional properties (formation energy, band gap, elasticity):

1. Get API key from [Materials Project](https://materialsproject.org/api)
2. Set environment variable: `export MP_API_KEY="your_key"`
3. Install: `pip install mp-api>=0.40.0`

Without API key, additional properties will be `None` (structures still available).

## Usage

### Generate Splits
```bash
python data_spilit.py
```

### Load Data
```python
import pickle
import pandas as pd

# Load any split
with open('processed_splits/random_split.pkl', 'rb') as f:
    data = pickle.load(f)

# Access features
X_train = data['X_train']
structures = X_train['structures']  # Pymatgen Structure objects
wyckoff = X_train['wyckoff_letters']
band_gaps = X_train['band_gap']  # None if no API key
df_X_train = pd.DataFrame(X_train)

# Access targets
y_train = data['y_train_log_klat']  # Log-transformed
y_train_original = data['y_train_klat']  # Original values
df_y_train = pd.DataFrame(y_train)
```

### Example: Using Structures
```python
# First training structure
structure = X_train['structures'][0]
print(structure.composition)
print(structure.lattice)
print(structure.density)
```

## Output Files

### Split Data
- `processed_splits/random_split.pkl`: Random 80/20 baseline split
- `processed_splits/space_group_split.pkl`: Space group disjoint split
- `processed_splits/ood_split.pkl`: Out-of-distribution split

### Visualizations

#### Space Group Split Visualization
![Space Group Split](figures/space_group_split_highlight.png)

Two horizontal rectangles showing:
- **Top**: Train space groups highlighted in color (grayed out = test only)
- **Bottom**: Test space groups highlighted in color (grayed out = train only)
- **X-axis**: Selected space group labels (showing ~20 labels to avoid overlap)
- Each colored segment represents a different space group, with width proportional to sample count
- Largest groups and evenly spaced groups are labeled for clarity

#### OOD klattice Histogram
![OOD Histogram](figures/ood_klat_histogram.png)

Overlaid histogram showing:
- **X-axis**: log(k_lattice) with 50 bins for better resolution
- **Blue**: Training set distribution (klat > 0.8)
- **Red**: Test set distribution (klat < 1.0)
- **Grid**: Added for easier reading
- Clearly shows test set concentrated at lower log(klat) values (< 0.4) and train set at higher values (> 0)

## Models

Custom models developed in this work are available in the corresponding branches of this repository.

- [ALIEGNN](https://github.com/dannyzekunren/Dataset_thermoconductivity_pred/tree/Zeyu)
- [Orb + {CNN, TabPFN}](https://github.com/dannyzekunren/Dataset_thermoconductivity_pred/tree/Danny)
- [MACE + MLP](https://github.com/dannyzekunren/Dataset_thermoconductivity_pred/tree/mahpe)
- [KAN, MLP, XGB, LR](https://github.com/dannyzekunren/Dataset_thermoconductivity_pred/tree/jianghai)
- [ViKING](https://github.com/dannyzekunren/Dataset_thermoconductivity_pred/tree/kedhip)

For the benchmark models used in the paper, please refer to their official repositories and follow the instructions provided there to reproduce the reported results:

- [HackNIP](https://github.com/parkyjmit/HackNIP)
- [CGCNN](https://github.com/txie-93/cgcnn)
- [WyFormer](https://github.com/SymmetryAdvantage/WyckoffTransformer)
- [CrabNet](https://github.com/anthony-wang/CrabNet)
- [eqV2-MEX](https://github.com/panmianzhi/Matching-based-EXtrapolation)

## Assessment Metrics

### Traditional Metrics
- **R² (Coefficient of Determination)**: Standard regression performance
- **MAE (Mean Absolute Error)**: Average absolute prediction error

### Specialized Metrics

#### Low-κ Weighted Log-MAE (κ-WLMAE)
A specialized metric designed for thermal conductivity prediction that emphasizes accuracy in the low thermal conductivity regime:

**Formula:**
```
κ-WLMAE = Σᵢ w(yᵢ) |log(ŷᵢ) - log(yᵢ)| / Σᵢ w(yᵢ)
```

**Weight Function:**
```
w(y) = min(1, (2/y)^p)
```

**Parameters:**
- **p = 2** (default): Mild-moderate emphasis on low-κ materials
- **p > 2**: Stronger focus on materials with κ < 2
- **log_base**: Natural log ('e') or base-10 ('10')

**Implementation:**
```python
import numpy as np

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
```

**Why κ-WLMAE:**
- **Unit-agnostic**: Multiplicative error measurement (|log ŷ - log y| ≈ relative error)
- **Low-κ focused**: Samples with κ ≤ 2 get full weight; above 2, weight decays smoothly
- **Robust**: Handles the wide range of thermal conductivity values 
- **Simple**: One hyperparameter (p) controls emphasis on low-κ materials
- **No discontinuities**: Smooth weight function, easy to implement and explain

## Key Features

- ✅ Three split strategies: random baseline, space group disjoint, OOD
- ✅ Comprehensive features: structures, Wyckoff positions, space groups
- ✅ Log-transformed targets with original values included
- ✅ Zero sample overlap across all splits
- ✅ Optional Materials Project API integration
- ✅ Pymatgen Structure objects for analysis
- ✅ Reproducible with fixed random seeds
- ✅ Specialized assessment metrics for thermal conductivity prediction

## Notes

- Original dataset had 200 duplicate mp_ids (handled appropriately)
- All splits ensure no sample overlap
- OOD split excludes 10 samples spanning thresholds
- Additional properties require MP API key

## Citation

To cite the archived dataset, use the [v1.0.0 Zenodo record](https://doi.org/10.5281/zenodo.22894537). The [concept DOI](https://doi.org/10.5281/zenodo.22894536) links to all versions.

If you use this dataset or the surrogate models in your work, please consider citing the following papers:

```bibtex
@article{Ohnishi2026,
  title = {Database and deep-learning scalability of anharmonic phonon properties by automated brute-force first-principles calculations},
  volume = {12},
  ISSN = {2057-3960},
  url = {http://dx.doi.org/10.1038/s41524-026-02033-w},
  DOI = {10.1038/s41524-026-02033-w},
  number = {1},
  journal = {npj Computational Materials},
  publisher = {Springer Science and Business Media LLC},
  author = {Ohnishi, Masato and Deng, Tianqi and Torres, Pol and Xu, Zhihao and Tadano, Terumasa and Zhang, Haoming and Nong, Wei and Hanai, Masatoshi and Wang, Zeyu and Morita, Michimasa and Tian, Zhiting and Hu, Ming and Ruan, Xiulin and Yoshida, Ryo and Suzumura, Toyotaro and Lindsay, Lucas and McGaughey, Alan J. H. and Luo, Tengfei and Hippalgaonkar, Kedar and Shiomi, Junichiro},
  year = {2026},
  month = apr
}

@misc{Wang2026,
  doi = {10.48550/ARXIV.2605.11610},
  url = {https://arxiv.org/abs/2605.11610},
  author = {Wang, Zeyu and Yamazaki, Shuya and Petersen, Martin Hoffmann and Ohnishi, Masato and Yamamoto, Tomiya and Nong, Wei and Wang, Jianghai and Zhu, Ruiming and Hanai, Masatoshi and Morita, Michimasa and Suzumura, Toyotaro and Ren, Zekun and Shiomi, Junichiro and Hippalgaonkar, Kedar},
  keywords = {Materials Science (cond-mat.mtrl-sci), Computational Physics (physics.comp-ph), FOS: Physical sciences, FOS: Physical sciences},
  title = {Fast and Accurate Prediction of Lattice Thermal Conductivity via Machine Learning Surrogates},
  publisher = {arXiv},
  year = {2026},
  copyright = {Creative Commons Attribution 4.0 International}
}

