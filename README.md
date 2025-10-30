# Forward Models for Thermal Conductivity Prediction

@[Jianghai](https://github.com/Ocean-JH)

This branch contains implementations of various forward models used to predict thermal conductivity in materials science.
Notes:
- It is recommended to run all scripts on GPU.
- All performance metrics are evaluated under <mark>log</mark> scale.
## Table of Contents

- [Features](#features)
- [Models Implemented](#models-implemented)
  - [XGBoost Regressor](#xgboost-regressor)
  - [Kolmogorov-Arnold Networks (KANs)](#kolmogorov-arnold-networks-kans)
  - [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
- [Evaluation](#evaluation)
- [Getting Started](#getting-started)
- [Project structure](#project-structure)
- [Scripts & Key Files](#scripts--key-files)
  - [Feature extraction](#feature-extraction)
  - [Training scripts](#training-scripts)
  - [Saved models](#saved-models)
- [TODO](#todo)

### Features
All models share the same feature set for consistency in comparison. The features include:
- matminer composition features
- SOAP structure features
- Space group features (SymmCD representation)

The feature extraction is done by the script: `feature/featurization.py`.

### Models Implemented
1. XGBoost Regressor
   - Implemented in `train/xgboost/train_xbg.py`
   - Parameters were optimized by grid search.
   - Performance:
   
     | Metric | Random | SPG | OOD |
     |-------|-------|------|------|
     | MAE | 0.444 | 0.540 | 1.610 |
     | $R^2$ | 0.758 | 0.705 | -6.695 |

2. Kolmogorov-Arnold Networks (KANs)
    - Implemented in `train/kan/train_kan.py` (from [EfficientKAN](https://github.com/Blealtan/efficient-kan))
    - Performance:
    
      | Metric | Random | SPG   | OOD    |
      |-------|--------|-------|--------|
      | MAE | 0.473  | 0.596 | 1.424  |
      | $R^2$ | 0.748   | 0.664 | -5.074 |

3. Multi-Layer Perceptron (MLP)
    - Implemented in `train/mlp/train_mlp.py`
    - Performance:
    
      | Metric | Random | SPG   | OOD    |
      |-------|--------|-------|--------|
      | MAE | 0.431  | 0.627 | 1.454  |
      | $R^2$ | 0.751  | 0.631 | -5.968 |


### Evaluation
To evaluate the models, run each script in the respective training folder to train a model from scratch. The evaluation metrics include Mean Absolute Error (MAE) and R-squared ($R^2$) on three different data splits: Random, Space Group (SPG), and Out-Of-Distribution (OOD). <mark>Remember to change the path when evaluating different split.</mark>

Alternatively, you can load models from the `models` directory to reproduce the results directly.

```python
from joblib import load

model = load("models/kan_random.pkl")
```

The MAE and $R^2$ can be computed as follows:

```python
import pickle
import torch
from joblib import load
from sklearn.metrics import mean_absolute_error, r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("dataset/featurized_data/random_scaled_full_soap_bin_matrix_sym_logk.pkl", "rb") as f:
    dataset = pickle.load(f)

X_train = dataset['train_input']
y_train = dataset['train_label']
X_test = dataset['test_input']
y_test = dataset['test_label']

model = load("models/kan_random.pkl")

# For KAN model prediction
y_pred = model(X_test)

# For XGBoost model prediction
# y_pred = model.predict(X_test)

# For MLP model prediction
# from train.mlp.train_mlp import predict
# y_pred = predict(model, X_test, log_transform=False)

if isinstance(X_test, torch.Tensor):
    X_test = X_test.detach().cpu().numpy()
if isinstance(y_test, torch.Tensor):
    y_test = y_test.detach().cpu().numpy()

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Final Test MAE (log y):", mae)
print("Final Test R² (log y):", r2)
```

### Getting Started

```bash
pip install -r requirements.txt
```

### Project structure
```
thermal-conductivity-prediction/
│
├── dataset/                   # Contains the dataset used for training and testing
│   ├── featurized_data/       # Featurized data files
│   └── raw_data/              # Raw data files
│
├── feature/                   # Feature extraction scripts
│   └── featurization.py       # Script for extracting features from the dataset
│
├── models/                    # Saved models
│   ├── kan_random.pkl         # KAN model trained on random data
│   ├── xgboost_random.pkl     # XGBoost model trained on random data
│   └── mlp_random.pkl         # MLP model trained on random data
│
├── scripts/                   # Miscellaneous scripts
│   └── evaluate.py            # Script for evaluating model performance
│
├── train/                     # Training scripts
│   ├── kan/                   # KAN training scripts
│   │   └── train_kan.py       # Main script for training KANs
│   │
│   ├── mlp/                   # MLP training scripts
│   │   └── train_mlp.py       # Main script for training MLPs
│   │
│   └── xgboost/               # XGBoost training scripts
│       └── train_xbg.py       # Main script for training XGBoost models
│
├── .gitignore                 # Git ignore file
├── README.md                  # Project README file
└── requirements.txt           # Python dependencies
```

### Scripts & Key Files
- **Feature extraction**: `feature/featurization.py`
- **Training scripts**:
  - XGBoost: `train/xgboost/train_xbg.py`
  - KAN: `train/kan/train_kan.py`
  - MLP: `train/mlp/train_mlp.py`
- **Saved models**: `models/` directory contains the trained models.

### TODO
- [x] Attention regressor.
- [ ] Multi feature attention.
- [ ] Graph KANs.
- [ ] KAN Transformer.