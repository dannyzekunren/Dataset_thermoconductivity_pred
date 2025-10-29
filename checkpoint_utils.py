"""
Model checkpoint utilities for saving and loading trained models
"""

import torch
import json
from pathlib import Path

def save_model_checkpoint(model, optimizer, config, results, epoch, save_path):
    """
    Save complete model checkpoint with configuration and results
    
    Args:
        model: Trained PyTorch model
        optimizer: Model optimizer
        config: Model configuration dictionary
        results: Training/validation results
        epoch: Current epoch number
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'results': results,
        'model_architecture': str(model)
    }
    
    torch.save(checkpoint, save_path)
    print(f"✅ Model checkpoint saved to: {save_path}")
    
    # Also save configuration separately
    config_path = save_path.replace('.pth', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return save_path

def load_model_checkpoint(checkpoint_path, model_class, device):
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_class: Model class to instantiate
        device: Device to load model on
    
    Returns:
        model: Loaded model
        config: Model configuration
        results: Training results
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    
    # Instantiate model with config
    model = model_class(**config['model_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    print(f"📊 Epoch: {checkpoint['epoch']}")
    
    return model, config, checkpoint['results']

def create_model_card(config, results, save_path):
    """
    Create a model card with configuration and performance details
    """
    model_card = f"""# Attention Hybrid BNN Model Card

## Model Configuration
- **Architecture**: Attention Hybrid Bayesian Neural Network
- **Hidden Dimensions**: {config['best_hyperparameters']['hidden_dim']}
- **Embedding Dimensions**: {config['best_hyperparameters']['embedding_dim']}
- **Attention Heads**: {config['best_hyperparameters']['num_heads']}
- **Dropout Rate**: {config['best_hyperparameters']['dropout']}
- **Learning Rate**: {config['best_hyperparameters']['learning_rate']}
- **Weight Decay**: {config['best_hyperparameters']['weight_decay']}
- **Batch Size**: {config['best_hyperparameters']['batch_size']}

## Performance Results
- **Random Split R²**: {results['random_split']['r2']:.4f}
- **Random Split MAE**: {results['random_split']['mae']:.4f}
- **Space Group Split R²**: {results['space_group_split']['r2']:.4f}
- **Space Group Split MAE**: {results['space_group_split']['mae']:.4f}
- **OOD Split R²**: {results['ood_split']['r2']:.4f}
- **OOD Split MAE**: {results['ood_split']['mae']:.4f}
- **Average MAE**: {results['average_mae']:.4f}

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
- **Optimization Date**: {config['optimization_date']}
- **Device**: {config['best_hyperparameters']['device']}
- **Model Type**: {config['model_type']}
"""
    
    with open(save_path, 'w') as f:
        f.write(model_card)
    
    print(f"📄 Model card saved to: {save_path}")
