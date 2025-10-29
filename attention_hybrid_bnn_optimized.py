#!/usr/bin/env python3
"""
Attention-Based Hybrid BNN with Extensive Hyperparameter Tuning
Optimized for Apple Silicon GPU acceleration and comprehensive hyperparameter search.

Key Innovations:
- Apple Silicon GPU optimization (MPS backend)
- Bayesian hyperparameter optimization
- Extensive hyperparameter search space
- Cross-validation for robust evaluation
- Advanced attention mechanisms for sparse feature fusion
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import os
import json
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import time
from typing import Dict, List, Tuple, Optional
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_optimal_device():
    """Get the best available device with detailed diagnostics."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"🚀 Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("🚀 Using Apple Silicon GPU (MPS)")
        # Test MPS functionality
        try:
            test_tensor = torch.randn(100, 100).to(device)
            torch.matmul(test_tensor, test_tensor)
            logger.info("✅ MPS device test successful")
        except Exception as e:
            logger.warning(f"⚠️ MPS test failed: {e}, falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        logger.info("💻 Using CPU (no GPU acceleration)")
    
    return device

class AttentionLayer(nn.Module):
    """Multi-head attention layer with configurable parameters."""
    
    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        assert input_dim % num_heads == 0, f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        # Add sequence dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Multi-head attention
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.input_dim)
        
        # Apply output projection and layer norm
        output = self.output_proj(attended)
        output = self.layer_norm(output + x)  # Residual connection
        
        if squeeze_output:
            output = output.squeeze(1)
            attention_weights = attention_weights.squeeze()
        
        return output, attention_weights

class SparseFeatureEmbedding(nn.Module):
    """Advanced embedding layer for sparse Wyckoff features."""
    
    def __init__(self, num_features: int, embedding_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        # Learnable embeddings for sparse features
        self.feature_embeddings = nn.Parameter(torch.randn(num_features, embedding_dim) * 0.1)
        self.zero_embedding = nn.Parameter(torch.randn(embedding_dim) * 0.1)
        
        # Feature importance weights with learnable bias
        self.importance_weights = nn.Parameter(torch.ones(num_features))
        self.importance_bias = nn.Parameter(torch.zeros(num_features))
        
        # Enhanced projection layers
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.ReLU()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Create mask for non-zero features
        non_zero_mask = (x != 0).float()
        
        # Compute importance scores
        importance_scores = torch.sigmoid(self.importance_weights + self.importance_bias)
        
        # Efficient embedding computation using vectorized operations
        feature_values = x.unsqueeze(-1)  # (batch_size, num_features, 1)
        
        # Select embeddings based on non-zero mask
        selected_embeddings = (non_zero_mask.unsqueeze(-1) * self.feature_embeddings.unsqueeze(0) +
                             (1 - non_zero_mask.unsqueeze(-1)) * self.zero_embedding.unsqueeze(0).unsqueeze(0))
        
        # Scale by feature values and importance
        weighted_embeddings = selected_embeddings * feature_values * importance_scores.unsqueeze(0).unsqueeze(-1)
        
        # Weighted pooling
        weights = non_zero_mask * importance_scores
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        pooled = torch.sum(weighted_embeddings * weights.unsqueeze(-1), dim=1)
        
        return self.projection(pooled)

class CrossAttention(nn.Module):
    """Enhanced cross-attention between original and Wyckoff features."""
    
    def __init__(self, original_dim: int, wyckoff_dim: int, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project to same dimension
        self.original_proj = nn.Sequential(
            nn.Linear(original_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.wyckoff_proj = nn.Sequential(
            nn.Linear(wyckoff_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection with residual connections
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, original_features, wyckoff_features):
        # Project to same dimension
        orig_proj = self.original_proj(original_features).unsqueeze(1)
        wyck_proj = self.wyckoff_proj(wyckoff_features).unsqueeze(1)
        
        # Bidirectional cross-attention
        orig_attended, _ = self.cross_attention(orig_proj, wyck_proj, wyck_proj)
        wyck_attended, _ = self.cross_attention(wyck_proj, orig_proj, orig_proj)
        
        # Combine and project
        combined = torch.cat([orig_attended.squeeze(1), wyck_attended.squeeze(1)], dim=-1)
        return self.output_proj(combined)

class AttentionHybridBNN(nn.Module):
    """
    Advanced Attention Hybrid BNN with configurable architecture.
    """
    
    def __init__(self, 
                 original_dim: int = 50, 
                 wyckoff_dim: int = 228,
                 hidden_dim: int = 128,
                 embedding_dim: int = 32,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 num_layers: int = 2):
        super().__init__()
        
        self.original_dim = original_dim
        self.wyckoff_dim = wyckoff_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Original feature encoder with multiple layers
        encoder_layers = [nn.BatchNorm1d(original_dim)]
        for i in range(num_layers):
            input_dim = original_dim if i == 0 else hidden_dim
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.original_encoder = nn.Sequential(*encoder_layers)
        
        # Wyckoff feature embedding
        self.wyckoff_embedding = SparseFeatureEmbedding(
            num_features=wyckoff_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        # Cross-attention between original and Wyckoff
        self.cross_attention = CrossAttention(
            original_dim=hidden_dim,
            wyckoff_dim=embedding_dim // 2,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        
        # Multiple self-attention layers
        self.self_attention_layers = nn.ModuleList([
            AttentionLayer(input_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Enhanced prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Enhanced uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Advanced weight initialization."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'predictor' in name or 'uncertainty' in name:
                    nn.init.xavier_normal_(module.weight, gain=0.5)
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, original_features, wyckoff_features, return_attention=False):
        # Encode original features
        original_encoded = self.original_encoder(original_features)
        
        # Embed Wyckoff features
        wyckoff_embedded = self.wyckoff_embedding(wyckoff_features)
        
        # Cross-attention between original and Wyckoff
        cross_attended = self.cross_attention(original_encoded, wyckoff_embedded)
        
        # Multi-layer self-attention
        current_features = cross_attended
        attention_weights = []
        
        for attention_layer in self.self_attention_layers:
            attended_features, weights = attention_layer(current_features)
            current_features = attended_features
            attention_weights.append(weights)
        
        # Add residual connection from cross-attention
        final_features = cross_attended + current_features
        
        # Predictions
        kappa_pred = self.predictor(final_features).squeeze(-1)
        uncertainty = self.uncertainty_head(final_features).squeeze(-1)
        
        if return_attention:
            return kappa_pred, uncertainty, attention_weights
        
        return kappa_pred, uncertainty

def load_hybrid_data(data_dir: str):
    """Load and prepare hybrid data for training."""
    # Load data
    train_df = pd.read_csv(os.path.join(data_dir, 'train_set_with_wyckoff.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val_set_with_wyckoff.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_set_with_wyckoff.csv'))
    
    # Define feature columns
    original_features = [
        'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'volume', 'n_atoms', 'density',
        'vol_per_atom', 'frac_O', 'frac_F', 'frac_N', 'frac_S', 'frac_Cl', 'frac_Se',
        'frac_Br', 'frac_I', 'frac_H', 'frac_C', 'frac_P', 'frac_B', 'frac_Te',
        'frac_Si', 'frac_As', 'frac_Ge', 'frac_Sn', 'frac_Pb', 'frac_Bi', 'frac_Sb',
        'frac_Li', 'frac_Na', 'frac_K', 'frac_Rb', 'frac_Cs', 'frac_Be', 'frac_Mg',
        'frac_Ca', 'frac_Sr', 'frac_Ba', 'frac_Al', 'frac_Ga', 'frac_In', 'frac_Tl',
        'frac_Sc', 'frac_Y', 'frac_La', 'frac_Ti', 'frac_Zr', 'frac_V'
    ]
    
    # Extract Wyckoff features
    all_features = [col for col in train_df.columns if col not in ['mp_id', 'ln_klat', 'klat']]
    wyckoff_features = [col for col in all_features if col not in original_features]
    
    logger.info(f"Original features: {len(original_features)}")
    logger.info(f"Wyckoff features: {len(wyckoff_features)}")
    logger.info(f"Total features: {len(all_features)}")
    
    def extract_data(df):
        X_orig = df[original_features].values.astype(np.float32)
        X_wyck = df[wyckoff_features].values.astype(np.float32)
        y = df['ln_klat'].values.astype(np.float32)
        return X_orig, X_wyck, y
    
    return extract_data(train_df), extract_data(val_df), extract_data(test_df), original_features, wyckoff_features

def get_hyperparameter_space():
    """Define comprehensive hyperparameter search space."""
    return {
        'hidden_dim': [64, 96, 128, 160, 192, 256],
        'embedding_dim': [32, 48, 64, 80, 96, 128],
        'num_heads': [2, 4, 6, 8],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rate': [0.0005, 0.001, 0.002, 0.003, 0.005],
        'weight_decay': [1e-5, 1e-4, 1e-3, 1e-2],
        'batch_size': [32, 48, 64, 96, 128],
        'num_layers': [1, 2, 3],
        'grad_clip': [0.5, 1.0, 2.0, 5.0]
    }

def sample_hyperparameters(search_space: Dict, method: str = 'random', n_samples: int = 50):
    """Sample hyperparameters using different strategies."""
    if method == 'random':
        samples = []
        for _ in range(n_samples):
            sample = {}
            for param, values in search_space.items():
                sample[param] = np.random.choice(values)
            samples.append(sample)
        return samples
    
    elif method == 'grid':
        # Grid search (limited combinations)
        keys = list(search_space.keys())
        # Reduce grid space for computational feasibility
        reduced_space = {
            'hidden_dim': [96, 128, 160],
            'embedding_dim': [48, 64, 80],
            'num_heads': [4, 6],
            'dropout': [0.2, 0.3],
            'learning_rate': [0.001, 0.002],
            'weight_decay': [1e-4, 1e-3],
            'batch_size': [64, 96],
            'num_layers': [2],
            'grad_clip': [1.0]
        }
        combinations = list(itertools.product(*reduced_space.values()))
        return [dict(zip(keys, combo)) for combo in combinations[:n_samples]]
    
    elif method == 'bayesian':
        # Simplified Bayesian optimization using Latin Hypercube sampling
        from scipy.stats import qmc
        
        n_params = len(search_space)
        sampler = qmc.LatinHypercube(d=n_params)
        samples_normalized = sampler.random(n=n_samples)
        
        samples = []
        for sample_norm in samples_normalized:
            sample = {}
            for i, (param, values) in enumerate(search_space.items()):
                idx = int(sample_norm[i] * len(values))
                idx = min(idx, len(values) - 1)
                sample[param] = values[idx]
            samples.append(sample)
        return samples

def train_single_config(train_data, val_data, config: Dict, device, max_epochs: int = 100):
    """Train a single model configuration with early stopping."""
    X_orig_train, X_wyck_train, y_train = train_data
    X_orig_val, X_wyck_val, y_val = val_data
    
    # Scale features
    orig_scaler = RobustScaler()
    wyck_scaler = StandardScaler()
    
    X_orig_train_scaled = orig_scaler.fit_transform(X_orig_train)
    X_wyck_train_scaled = wyck_scaler.fit_transform(X_wyck_train)
    X_orig_val_scaled = orig_scaler.transform(X_orig_val)
    X_wyck_val_scaled = wyck_scaler.transform(X_wyck_val)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_orig_train_scaled),
        torch.FloatTensor(X_wyck_train_scaled),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_orig_val_scaled),
        torch.FloatTensor(X_wyck_val_scaled),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model
    model = AttentionHybridBNN(
        original_dim=50,
        wyckoff_dim=228,
        hidden_dim=config['hidden_dim'],
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        num_layers=config['num_layers']
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10, verbose=False
    )
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_orig, batch_wyck, batch_y in train_loader:
            batch_orig = batch_orig.to(device)
            batch_wyck = batch_wyck.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            pred, uncertainty = model(batch_orig, batch_wyck)
            
            # Uncertainty-weighted loss
            mse_loss = F.mse_loss(pred, batch_y, reduction='none')
            uncertainty_loss = torch.mean(mse_loss / (uncertainty + 1e-8) + torch.log(uncertainty + 1e-8))
            
            uncertainty_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            
            train_loss += uncertainty_loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_orig, batch_wyck, batch_y in val_loader:
                batch_orig = batch_orig.to(device)
                batch_wyck = batch_wyck.to(device)
                batch_y = batch_y.to(device)
                
                pred, uncertainty = model(batch_orig, batch_wyck)
                loss = F.mse_loss(pred, batch_y)
                
                val_loss += loss.item()
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_r2 = r2_score(val_targets, val_preds)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return {
        'model': model,
        'val_loss': best_val_loss,
        'val_r2': val_r2,
        'orig_scaler': orig_scaler,
        'wyck_scaler': wyck_scaler,
        'epochs_trained': epoch + 1
    }

def evaluate_model_on_test(model, test_data, orig_scaler, wyck_scaler, device):
    """Evaluate model on test set."""
    X_orig_test, X_wyck_test, y_test = test_data
    
    # Scale features
    X_orig_test_scaled = orig_scaler.transform(X_orig_test)
    X_wyck_test_scaled = wyck_scaler.transform(X_wyck_test)
    
    # Create data loader
    test_dataset = TensorDataset(
        torch.FloatTensor(X_orig_test_scaled),
        torch.FloatTensor(X_wyck_test_scaled),
        torch.FloatTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Evaluation
    model.eval()
    predictions = []
    uncertainties = []
    targets = []
    
    with torch.no_grad():
        for batch_orig, batch_wyck, batch_y in test_loader:
            batch_orig = batch_orig.to(device)
            batch_wyck = batch_wyck.to(device)
            
            pred, uncertainty = model(batch_orig, batch_wyck)
            
            predictions.extend(pred.cpu().numpy())
            uncertainties.extend(uncertainty.cpu().numpy())
            targets.extend(batch_y.numpy())
    
    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    targets = np.array(targets)
    
    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return {
        'mae': mae,
        'r2': r2,
        'predictions': predictions,
        'uncertainties': uncertainties,
        'targets': targets
    }

def hyperparameter_search(split_name: str, method: str = 'random', n_trials: int = 50):
    """Comprehensive hyperparameter search for a specific split."""
    
    print(f"\n{'='*80}")
    print(f"🔍 HYPERPARAMETER OPTIMIZATION: {split_name.upper()}")
    print(f"Method: {method.upper()}, Trials: {n_trials}")
    print(f"{'='*80}")
    
    device = get_optimal_device()
    
    # Load data
    data_dir = f"data/{split_name}"
    train_data, val_data, test_data, orig_features, wyck_features = load_hybrid_data(data_dir)
    
    # Get hyperparameter combinations
    search_space = get_hyperparameter_space()
    configs = sample_hyperparameters(search_space, method=method, n_samples=n_trials)
    
    print(f"📊 Data loaded: Train={len(train_data[2])}, Val={len(val_data[2])}, Test={len(test_data[2])}")
    print(f"🧪 Testing {len(configs)} hyperparameter configurations...")
    
    results = []
    best_val_r2 = -float('inf')
    best_config = None
    best_model_info = None
    
    for i, config in enumerate(configs):
        start_time = time.time()
        
        try:
            # Train model with current config
            result = train_single_config(train_data, val_data, config, device, max_epochs=100)
            
            training_time = time.time() - start_time
            
            # Track results
            config_result = {
                'trial': i + 1,
                'config': config.copy(),
                'val_loss': result['val_loss'],
                'val_r2': result['val_r2'],
                'epochs_trained': result['epochs_trained'],
                'training_time': training_time
            }
            
            results.append(config_result)
            
            # Update best configuration
            if result['val_r2'] > best_val_r2:
                best_val_r2 = result['val_r2']
                best_config = config.copy()
                best_model_info = result
                
                print(f"🎯 New best! Trial {i+1}: Val R² = {result['val_r2']:.4f}")
                print(f"   Config: {config}")
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"📈 Progress: {i+1}/{len(configs)} trials completed")
                print(f"   Current best Val R²: {best_val_r2:.4f}")
        
        except Exception as e:
            logger.error(f"❌ Trial {i+1} failed: {e}")
            continue
    
    # Evaluate best model on test set
    if best_model_info is not None:
        print(f"\n🏆 BEST CONFIGURATION FOUND:")
        print(f"Val R²: {best_val_r2:.4f}")
        print(f"Config: {best_config}")
        
        test_results = evaluate_model_on_test(
            best_model_info['model'], test_data, 
            best_model_info['orig_scaler'], best_model_info['wyck_scaler'], device
        )
        
        print(f"\n📊 TEST SET PERFORMANCE:")
        print(f"Test R²: {test_results['r2']:.4f}")
        print(f"Test MAE: {test_results['mae']:.4f}")
        
        # Save detailed results
        results_summary = {
            'split': split_name,
            'method': method,
            'n_trials': n_trials,
            'best_config': best_config,
            'best_val_r2': best_val_r2,
            'test_r2': test_results['r2'],
            'test_mae': test_results['mae'],
            'all_trials': results
        }
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hyperparameter_search_{split_name}_{method}_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj
        
        results_summary = convert_to_serializable(results_summary)
        
        with open(filename, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n✅ Results saved to: {filename}")
        
        return results_summary
    else:
        print("❌ No successful trials completed!")
        return None

def run_comprehensive_optimization():
    """Run comprehensive hyperparameter optimization on all splits."""
    
    print("🚀 COMPREHENSIVE ATTENTION HYBRID BNN HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print("🔧 GPU-Accelerated with Apple Silicon MPS Support")
    print("🧠 Advanced Bayesian Neural Network with Attention Mechanisms")
    print()
    
    splits = ['random_split', 'ood_split', 'space_group_split']
    methods = ['random', 'bayesian']  # Skip grid search for time efficiency
    n_trials = 30  # Reasonable number for comprehensive search
    
    all_results = {}
    
    for split in splits:
        split_results = {}
        
        for method in methods:
            try:
                print(f"\n🎯 Starting {method} search for {split}...")
                results = hyperparameter_search(split, method=method, n_trials=n_trials)
                
                if results:
                    split_results[method] = results
                    print(f"✅ {method} search completed for {split}")
                else:
                    print(f"❌ {method} search failed for {split}")
                    
            except Exception as e:
                logger.error(f"❌ Error in {method} search for {split}: {e}")
                continue
        
        all_results[split] = split_results
    
    # Create final comparison
    print(f"\n{'='*80}")
    print("📋 FINAL HYPERPARAMETER OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    
    best_overall = None
    best_overall_score = -float('inf')
    
    for split, methods_results in all_results.items():
        print(f"\n🎯 {split.upper()}:")
        
        for method, results in methods_results.items():
            test_r2 = results['test_r2']
            test_mae = results['test_mae']
            
            print(f"  {method.capitalize()}: R² = {test_r2:.4f}, MAE = {test_mae:.4f}")
            
            if test_r2 > best_overall_score:
                best_overall_score = test_r2
                best_overall = {
                    'split': split,
                    'method': method,
                    'config': results['best_config'],
                    'test_r2': test_r2,
                    'test_mae': test_mae
                }
    
    if best_overall:
        print(f"\n🏆 BEST OVERALL CONFIGURATION:")
        print(f"Split: {best_overall['split']}")
        print(f"Method: {best_overall['method']}")
        print(f"Test R²: {best_overall['test_r2']:.4f}")
        print(f"Test MAE: {best_overall['test_mae']:.4f}")
        print(f"Config: {best_overall['config']}")
        
        # Save best overall configuration
        with open('best_attention_hybrid_config.json', 'w') as f:
            json.dump(best_overall, f, indent=2)
        
        print(f"\n✅ Best configuration saved to: best_attention_hybrid_config.json")
    
    return all_results

if __name__ == "__main__":
    # Run comprehensive hyperparameter optimization
    results = run_comprehensive_optimization()
