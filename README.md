# ALIEGNN: Atomistic Line Graph Equivariant Neural Network

A deep learning model for predicting material properties based on atomic structures. ALIEGNN incorporates equivariant neural network principles with line graph representations for improved performance on crystalline materials.

## Model Performance

| Split Type | MAE ↓ | R² ↑ |
|------------|-------|------|
| Random | 0.379 | 0.762 |
| Space Group | 0.512 | 0.697 |
| OOD | 1.245 | -3.595 |
| **Average MAE** | **0.712** | - |

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Prediction](#prediction)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.8 or later
- CUDA 11.8 (for GPU support)
- Conda or Miniconda

### Step 1: Create Conda Environment

```bash
# Create and activate environment
conda create -n aliegnn python=3.8 -y
conda activate aliegnn

# Load CUDA modules (if on HPC cluster)
module load cuda/11.8
module load nvhpc/25.3
```

### Step 2: Install PyTorch with CUDA Support

**IMPORTANT:** Install PyTorch BEFORE installing ALIEGNN.

```bash
# Install uv package manager
pip install uv

# Install PyTorch with CUDA 11.8
uv pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install PyG extensions
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
```

### Step 3: Install DGL with CUDA Support

```bash
conda install -c dglteam/label/th24_cu118 dgl -y
```

### Step 4: Install ALIEGNN Package

```bash
# Navigate to project directory
cd /path/to/Dataset_thermoconductivity_pred

# Install in editable mode
uv pip install -e .
```

### Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Check DGL
python -c "import dgl; print('DGL:', dgl.__version__)"

# Check ALIEGNN
python -c "from aliegnn.config import TrainingConfig; from aliegnn.models.alignn_egnn import ALIEGNN; print('✓ ALIEGNN installed successfully!')"
```

### CPU-Only Installation (Optional)

If you don't have a GPU:

```bash
pip install torch torchvision
conda install -c dglteam dgl
pip install -e .
```

---

## Quick Start

### Dataset Preparation

1. **Obtain dataset files:**

```bash
python tools/data_obtain.py \
    --source_file_path ./processed_splits/random_split.pkl \
    --target_file_path ./processed_splits/ \
    --target_name random
```

2. **Convert structures to POSCAR:**

```bash
python tools/struc2poscar.py \
    --source_file_path ./processed_splits/random_split.pkl \
    --target_file_path ./processed_splits/
```
3. **Move the config file to target directory:**

Expected directory structure:
```
processed_splits/
├── config_example.json
├── random_log_train_data.csv
├── random_log_test_data.csv
├── mp-226 (POSCAR)/
├── mp-361 (POSCAR)/
└── ...
```

---

## Training

### Train Model

```bash
python aliegnn/script/train_folder_split_data.py \
    --root_dir ./processed_splits \
    --config_name config_example.json \
    --id_prop_file_train random_train_data.csv \
    --id_prop_file_test random_test_data.csv \
    --output_dir ./output_random_split \
```

### Training Arguments

- `--root_dir`: Directory containing structure files and CSV data
- `--config_name`: Model configuration file (JSON)
- `--id_prop_file_train`: Training data CSV (format: `mp-id,target_value`)
- `--id_prop_file_test`: Test data CSV
- `--output_dir`: Directory for saving trained model and logs

### Training Outputs

The training script creates:
```
output_random_split/
├── best_model.pt                            # Best model checkpoint
├── checkpoint_model.pt                      # Latest checkpoint
├── config.json                              # Model configuration
├── history_train.json                       # Training history
├── prediction_metrics_results.csv           # Metrics
├── prediction_results_test_set.csv          # Predictions for test set
└── ...
```

---

## Prediction

### Simple Prediction Script

Use `simple_predict.py` to directly load a trained `.pt` model and make predictions.

### Single Structure Prediction

```bash
python aliegnn/script/simple_predict.py \
    --model output_random_split/best_model.pt \
    --config output_random_split/config.json \
    --structure path/to/POSCAR
```

### Batch Prediction from CSV (Recommended)

```bash
python aliegnn/script/simple_predict.py \
    --model output_random_split/best_model.pt \
    --config output_random_split/config.json \
    --csv processed_splits/random_test_data.csv \
    --output test_predictions
```

**CSV Format:**
```csv
mp-226,4.0514321184272095
mp-361,1.0534837971018483
mp-380,2.8814006807614914
```
- First column: Material ID (mp-id)
- Second column: Ground truth value (optional, for error calculation)

### Prediction Outputs

The script creates a directory with three files:

```
test_predictions/
├── predictions.csv      # Detailed predictions for each material
├── predictions.json     # Same data in JSON format
└── SUMMARY.txt          # Overall statistics and metrics
```

**predictions.csv columns:**
- `mp_id`: Material identifier
- `formula`: Chemical formula
- `ground_truth`: True value (if provided in input CSV)
- `prediction`: Model prediction
- `absolute_error`: |prediction - ground_truth|
- `relative_error_percent`: Relative error as percentage

**SUMMARY.txt includes:**
- Input file paths
- Total/Success/Failed counts
- Prediction statistics (Mean, Std, Min, Max)
- Error metrics: MAE, R², MAPE (if ground truth available)
- List of failed materials

### Supported File Formats

- **POSCAR/CONTCAR**: VASP structure files
- **CIF**: Crystallographic Information File
- **XYZ**: XYZ coordinate file

The script auto-detects file format based on extension.

### Advanced Options

```bash
python aliegnn/script/simple_predict.py \
    --model best_model.pt \
    --config config.json \
    --csv test_data.csv \
    --structures_dir /custom/path/to/structures \  # Custom structure directory
    --output my_predictions \                       # Custom output directory
    --format cif                                    # Force file format
```

---

## Troubleshooting

### CUDA Version Mismatch

Check your CUDA version:
```bash
nvcc --version
nvidia-smi
```

Install matching PyTorch and DGL versions for your CUDA version.

### Import Errors

If you encounter import errors after installation:
```bash
# Reinstall ALIEGNN
pip install -e . --force-reinstall --no-deps
```

### Memory Issues

If you encounter out-of-memory errors during training:
- Reduce batch size: `--batch_size 32`
- Use gradient accumulation
- Monitor GPU usage: `watch -n 1 nvidia-smi`

### Structure File Not Found

When using CSV prediction, ensure:
1. Structures are in the same directory as CSV, or use `--structures_dir`
2. Structure folders are named exactly as mp-id in CSV
3. POSCAR/CONTCAR files exist in each structure folder

---

## Dependencies

### Core Requirements

- Python: 3.8+
- PyTorch: 2.4.1+cu118
- Torchvision: 0.19.1+cu118
- DGL: 2.4.0 (th24_cu118)
- CUDA: 11.8

### Additional Dependencies

- jarvis-tools
- numpy
- pandas
- tqdm
- pydantic

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: WANG ZEYU

## 🔧 Required Patch: Jarvis

This project requires a small modification to the Jarvis source code.
Or you will meet a issue:
```
line 688, in __getitem__
    return self._columns[name].data
KeyError: 'coords'
```

Please edit the function `Graph.atom_dgl_multigraph` in:
/path/to/envs/aliegnn/lib/python3.8/site-packages/jarvis/core/graphs.py

```python
g.ndata["atom_features"] = node_features
g.edata["r"] = r
g.ndata["frac_coords"] = torch.tensor(atoms.frac_coords).type(
    torch.get_default_dtype()
)
g.ndata["coords"] = torch.tensor(atoms.cart_coords).type(
    torch.get_default_dtype()
)
```
---

## Acknowledgments

Based on the ALIGNN architecture with equivariant neural network enhancements.
