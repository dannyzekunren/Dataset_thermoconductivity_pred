# Thermal Conductivity Dataset Splits

This repository contains three distinct train/test splits for thermal conductivity prediction models, generated from Materials Project data.

## ⭐ Best Model: CNN-B

**CNN-B (sequence embeddings) is the recommended model** for thermal conductivity prediction. It processes per-atom embeddings with masked global pooling, achieving superior performance across all data splits (random, space group, and out-of-distribution splits). CNN-B effectively handles variable-length crystal structures and captures atomic-level patterns that are critical for thermal conductivity prediction.

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

### 1. Install Base Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install PyTorch

**Important**: PyTorch installation depends on your CUDA version. Install the appropriate version:

- **With CUDA support**: Visit [PyTorch installation page](https://pytorch.org/get-started/locally/) and select your CUDA version
- **CPU only**: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

**Note**: Ensure PyTorch CUDA version matches your GPU driver. If you encounter CUDA errors, see the Troubleshooting section.

### 3. Install ORB Models (Required for Embedding Extraction)

The script uses `orb-models` to extract embeddings from crystal structures. Install it using:

```bash
pip install orb-models
```

**Note**: The ORB model (`orb-v3-conservative-20-omat`) will be automatically downloaded on first use. This requires internet connection and may take several minutes depending on your connection speed.

### 4. Optional: Install Additional Model Dependencies

#### For CGCNN (Graph Neural Network):
```bash
pip install torch-geometric
```

#### For TabPFN (Tabular Transformer):
```bash
pip install tabpfn
```

**Note**: Both CGCNN and TabPFN are optional. If not installed, the script will skip these models with a warning.

### 5. Optional: Materials Project API

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

### Train Models with MLIP Embeddings

The `embedding_regression.py` script extracts embeddings from ORB models and trains various architectures (CNN-A, CNN-B, CGCNN, TabPFN) to predict thermal conductivity.

#### Quick Start: Train CNN-B on All Splits (Recommended)

The simplest command to train CNN-B (the best model) on all three splits:

```bash
python embedding_regression.py --model CNN-B
```

**Note**: The script now automatically detects CUDA compatibility issues and falls back to CPU mode if needed. No additional wrapper scripts are required for most users.

Or explicitly specify all splits:

```bash
python embedding_regression.py --model CNN-B --splits ood_split random_split space_group_split
```

#### CLI Command Reference

**Basic Usage:**
```bash
python embedding_regression.py [OPTIONS]
```

**Available Options:**

- `--model`: Model to train
  - `CNN-A`: Aggregated embeddings (mean + max pooling)
  - `CNN-B`: Per-atom sequence embeddings with masked pooling  **Best**
  - `CGCNN`: Crystal Graph Convolutional Neural Network (requires `torch-geometric`)
  - `TabPFN`: Tabular transformer using masked statistics (requires `tabpfn`)
  - `all`: Train all available models (default)

- `--splits`: Data splits to process (space-separated)
  - `ood_split`: Out-of-distribution split
  - `random_split`: Random 80/20 split
  - `space_group_split`: Space group disjoint split
  - Default: all three splits

- `--use-properties`: Enable material properties as additional features
  - Includes: formation energy, e_above_hull, band gap, elasticity_K_VRH, elasticity_G_VRH

- `--ood-scaler`: Scaler type for OOD split properties
  - `robust`: RobustScaler (default, handles outliers better)
  - `standard`: StandardScaler
  - `minmax`: MinMaxScaler

- `--weighted-loss`: Use weighted loss focusing on log(κ) < 1

#### Example Commands

**1. Train CNN-B on all splits (recommended):**
```bash
python embedding_regression.py --model CNN-B
```

**2. Train CNN-B with material properties:**
```bash
python embedding_regression.py --model CNN-B --use-properties
```

**3. Train CNN-A only:**
```bash
python embedding_regression.py --model CNN-A --splits random_split
```

**4. Train all models on OOD split:**
```bash
python embedding_regression.py --model all --splits ood_split
```

**5. Train CGCNN with properties:**
```bash
python embedding_regression.py --model CGCNN --use-properties
```

**6. Train TabPFN on all splits:**
```bash
python embedding_regression.py --model TabPFN
```

**7. Train CNN-B with weighted loss (focus on low κ):**
```bash
python embedding_regression.py --model CNN-B --weighted-loss
```

#### Output

The script generates:
- **Console output**: Training progress, validation metrics, and summary table
- **Cache files**: Extracted embeddings saved in `cache/` directory (reused on subsequent runs)
- **Visualization**: `thermal_conductivity_predictions.png` with scatter plots for all model/split combinations
- **Metrics**: R² and MAE scores printed for each model and split

#### Performance Notes

- **First run**: Embedding extraction takes time (cached for future runs)
- **GPU recommended**: Training is faster on CUDA-enabled GPUs
- **Memory**: CGCNN and TabPFN require more memory; TabPFN has sample/feature limits

#### Troubleshooting

**CUDA Compatibility Issues:**

The script now automatically detects CUDA compatibility issues and falls back to CPU mode. If you encounter CUDA errors, the script will:

1. **Automatically test CUDA** when it starts
2. **Fall back to CPU** if CUDA operations fail
3. **Continue execution** on CPU (slower but compatible)

For manual control, you can also:

1. **Force CPU mode** (if needed):
   ```bash
   # Set environment variable to use CPU
   set CUDA_VISIBLE_DEVICES=-1
   python embedding_regression.py --model CNN-B
   ```

2. **Reinstall PyTorch with correct CUDA version**:
   - Check your CUDA version: `nvidia-smi`
   - Install matching PyTorch: https://pytorch.org/get-started/locally/
   - Example for CUDA 11.8:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

3. **Use CPU-only PyTorch**:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

### Load Data (Python API)
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
