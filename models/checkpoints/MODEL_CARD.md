# Attention Hybrid BNN Model Card

## Model Configuration
- **Architecture**: Attention Hybrid Bayesian Neural Network
- **Hidden Dimensions**: 128
- **Embedding Dimensions**: 64
- **Attention Heads**: 4
- **Dropout Rate**: 0.3
- **Learning Rate**: 0.001
- **Weight Decay**: 0.0001
- **Batch Size**: 64

## Performance Results
- **Random Split R²**: 0.7460
- **Random Split MAE**: 0.4870
- **Space Group Split R²**: 0.6720
- **Space Group Split MAE**: 0.5970
- **OOD Split R²**: -4.1410
- **OOD Split MAE**: 1.3700
- **Average MAE**: 0.8180

## Usage
```python
from checkpoint_utils import load_model_checkpoint
from attention_hybrid_bnn_simple import AttentionHybridBNN

# Load model
model, config, results = load_model_checkpoint(
    'models/checkpoints/best_attention_hybrid_bnn.pth',
    AttentionHybridBNN,
    device='auto'
)

# Make predictions
predictions = model(data)
```

## Training Details
- **Optimization Date**: 2025-10-29T22:33:06.027813
- **Device**: mps
- **Model Type**: Attention Hybrid BNN
