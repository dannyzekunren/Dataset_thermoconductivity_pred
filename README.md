# Thermal Conductivity Dataset Splits

This repository contains three distinct train/test splits for thermal conductivity prediction models, generated from Materials Project data.

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

# Load any split
with open('processed_splits/random_split.pkl', 'rb') as f:
    data = pickle.load(f)

# Access features
X_train = data['X_train']
structures = X_train['structures']  # Pymatgen Structure objects
wyckoff = X_train['wyckoff_letters']
band_gaps = X_train['band_gap']  # None if no API key

# Access targets
y_train = data['y_train_log_klat']  # Log-transformed
y_train_original = data['y_train_klat']  # Original values
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

- `processed_splits/random_split.pkl`: Random baseline
- `processed_splits/space_group_split.pkl`: Space group disjoint
- `processed_splits/ood_split.pkl`: Out-of-distribution
- `figures/space_group_split_highlight.png`: Space group visualization
- `figures/ood_klat_histogram.png`: OOD histogram (log scale)

## Key Features

- ✅ Three split strategies: random baseline, space group disjoint, OOD
- ✅ Comprehensive features: structures, Wyckoff positions, space groups
- ✅ Log-transformed targets with original values included
- ✅ Zero sample overlap across all splits
- ✅ Optional Materials Project API integration
- ✅ Pymatgen Structure objects for analysis
- ✅ Reproducible with fixed random seeds

## Notes

- Original dataset had 200 duplicate mp_ids (handled appropriately)
- All splits ensure no sample overlap
- OOD split excludes 10 samples spanning thresholds
- Additional properties require MP API key
