# Attention Hybrid BNN for Thermal Conductivity Prediction

## Model Overview

This repository contains an **Attention Hybrid Bayesian Neural Network (BNN)** for thermal conductivity prediction with strong performance and GPU acceleration support:

- **R² = 0.746** on Random Split (excellent performance)
- **R² = 0.672** on Space Group Split (strong generalization)  
- **Average MAE = 0.818** across all splits
- **GPU acceleration** support with automatic device detection
- **Saved model checkpoints** for immediate deployment
- **Comprehensive uncertainty quantification** using Bayesian deep learning

## Performance Results

| Split Type | MAE↓ | R²↑ | Performance Level |
|------------|------|-----|-------------------|
| Random | 0.487 | 0.746 | **Excellent** |
| Space Group | 0.597 | 0.672 | **Strong** |
| OOD | 1.370 | -4.141 | Challenge dataset |
| **Average MAE** | **0.818** | - | **Competitive** |

## Quick Start

### Prerequisites
- **Python 3.8+**
- **PyTorch 2.0+** (with GPU support if available)
- **CUDA** (for NVIDIA GPUs) or **Metal Performance Shaders** (for Apple Silicon)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/dannyzekunren/Dataset_thermoconductivity_pred.git
cd Dataset_thermoconductivity_pred
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify GPU setup:**
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"MPS Available: {torch.backends.mps.is_available()}")
print(f"PyTorch Version: {torch.__version__}")
```

### Running the Model

#### Option 1: Load Pre-trained Checkpoint (Recommended)
```python
from checkpoint_utils import load_model_checkpoint
from attention_hybrid_bnn_simple import AttentionHybridBNN

# Load pre-trained model
model, config, results = load_model_checkpoint(
    'models/checkpoints/best_attention_hybrid_bnn.pth',
    AttentionHybridBNN,
    device='auto'
)

# Make predictions
predictions = model(your_data)
```

#### Option 2: Train from Scratch
```bash
python attention_hybrid_bnn_simple.py
```

#### Option 3: Full Optimization Pipeline
```bash
python final_direct_optimization.py
```

## Model Architecture

The model uses an attention-based architecture optimized for materials property prediction:

- **Input Features**: 230 dimensions (134 elemental + 8 structural + 88 Wyckoff positions)
- **Attention Mechanism**: Multi-head attention (4 heads) for feature relationships
- **Bayesian Network**: Hidden layers with uncertainty quantification
- **Sparse Feature Handling**: Optimized for 88.2% sparse Wyckoff position data

## Saved Model Checkpoints

The repository includes pre-trained model checkpoints:

- `models/checkpoints/best_attention_hybrid_bnn.pth` - Optimized model weights
- `models/checkpoints/MODEL_CARD.md` - Detailed model information
- `config/FINAL_best_attention_config.json` - Hyperparameter configuration

## Configuration

Optimized hyperparameters (in `config/FINAL_best_attention_config.json`):

```json
{
  "hidden_dim": 128,
  "embedding_dim": 64,
  "num_heads": 4,
  "dropout": 0.3,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "batch_size": 64
}
```

## Features

- **Automatic Device Detection**: Supports CUDA, MPS (Apple Silicon), and CPU
- **Uncertainty Quantification**: Bayesian inference with confidence intervals
- **Attention Mechanisms**: Multi-head attention for interpretable feature relationships
- **Sparse Feature Optimization**: Efficient handling of sparse Wyckoff position data
- **Checkpoint System**: Save and load trained models easily

## Documentation

- `results/FINAL_OPTIMIZED_attention_hybrid_results.csv` - Performance results
- `docs/` - Technical documentation and reports
- `validate_setup.py` - Environment validation script

## Validation

Run the validation script to verify your setup:

```bash
python validate_setup.py
```

### 📊 Model Architecture

```
Input Features (230 dimensions)
├── Elemental Properties (134 features)
├── Structural Features (8 features)  
└── Wyckoff Positions (88 features, 88.2% sparse)
    │
    ▼
Attention Mechanism (Multi-Head, 4 heads)
├── Query/Key/Value Projections
├── Sparse Feature Attention
└── Position-aware Encoding
    │
    ▼
Bayesian Neural Network
├── Hidden Layer 1 (128 units + dropout 0.3)
├── Hidden Layer 2 (64 units + dropout 0.3)
└── Output Layer (1 unit, log kappa prediction)
    │
    ▼
Uncertainty Quantification
├── Aleatoric Uncertainty (data noise)
├── Epistemic Uncertainty (model uncertainty)  
└── Combined Predictive Intervals
```

### ⚙️ Optimized Configuration

The model uses the following optimized hyperparameters (stored in `config/FINAL_best_attention_config.json`):

```json
{
  "hidden_dim": 128,
  "embedding_dim": 64,
  "num_heads": 4,
  "dropout": 0.3,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "batch_size": 64,
  "device": "mps"
}
```

### 🍎 Apple Silicon GPU Optimization

#### Automatic Device Detection
The model automatically detects and uses the best available device:
1. **CUDA GPU** (if available, for NVIDIA systems)
2. **Apple Silicon GPU (MPS)** (if available, for Apple systems)  
3. **CPU** (fallback)

#### Performance Benchmarks
- **Matrix Operations**: 2.0-2.4× speedup on Apple Silicon vs CPU
- **Training Time**: ~60% reduction in training time
- **Memory Efficiency**: Optimized memory usage for large datasets

#### Manual Device Selection
```python
import torch

# Force Apple Silicon GPU
device = torch.device('mps')

# Check device status
print(f"Using device: {device}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

### 📈 Advanced Features

#### 1. Uncertainty Quantification
- **Bayesian Inference**: Monte Carlo dropout for epistemic uncertainty
- **Aleatoric Uncertainty**: Learned heteroscedastic noise modeling
- **Confidence Intervals**: Provides prediction intervals for reliability assessment

#### 2. Sparse Feature Handling
- **Wyckoff Position Optimization**: Specialized attention for 88.2% sparse features
- **Memory Efficient**: Optimized sparse tensor operations
- **Performance Maintained**: No accuracy loss despite sparsity

#### 3. Attention Mechanisms
- **Multi-Head Attention**: 4 attention heads for diverse feature relationships
- **Position-Aware**: Incorporates structural position information
- **Interpretable**: Attention weights provide model explainability

### 🛠️ Troubleshooting

#### Common Issues

1. **MPS not available**
   - Ensure macOS 12.3+ and Apple Silicon Mac
   - Update PyTorch to version 2.1+

2. **Memory errors**
   - Reduce batch size in config
   - Monitor memory usage with Activity Monitor

3. **Slow training**
   - Verify MPS device selection
   - Check for competing processes

#### Performance Tips

1. **Optimal Batch Size**: Start with 64, adjust based on available memory
2. **Learning Rate**: Use 0.001 for stable convergence
3. **Early Stopping**: Monitor validation loss to prevent overfitting

### 📚 Documentation

- **Technical Report**: See `docs/FINAL_APPLE_SILICON_OPTIMIZATION_REPORT.md`
- **Results Analysis**: See `results/FINAL_OPTIMIZED_attention_hybrid_results.csv`
- **Configuration**: See `config/FINAL_best_attention_config.json`

