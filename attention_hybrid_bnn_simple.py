#!/usr/bin/env python3
"""
Attention-Based Hybrid BNN for Thermal Conductivity Prediction
Combines original 50 features with sparse Wyckoff features through attention mechanisms.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import os

class AttentionHybridBNN(nn.Module):
    """Simplified Attention-based Hybrid BNN for testing."""
    
    def __init__(self, original_dim=50, wyckoff_dim=228, hidden_dim=128):
        super().__init__()
        
        # Original feature encoder 
        self.original_encoder = nn.Sequential(
            nn.Linear(original_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Wyckoff feature encoder with sparsity handling
        self.wyckoff_encoder = nn.Sequential(
            nn.Linear(wyckoff_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Attention layer for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        
        # Final predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, original_features, wyckoff_features):
        # Encode both feature types
        orig_encoded = self.original_encoder(original_features).unsqueeze(1)  # (B, 1, H)
        wyck_encoded = self.wyckoff_encoder(wyckoff_features).unsqueeze(1)    # (B, 1, H)
        
        # Apply attention
        orig_attended, _ = self.attention(orig_encoded, wyck_encoded, wyck_encoded)
        wyck_attended, _ = self.attention(wyck_encoded, orig_encoded, orig_encoded)
        
        # Combine and predict
        combined = torch.cat([orig_attended.squeeze(1), wyck_attended.squeeze(1)], dim=-1)
        return self.predictor(combined).squeeze(-1)

def load_hybrid_data(data_dir):
    """Load data and split into original and Wyckoff features."""
    
    # Load data
    train_df = pd.read_csv(os.path.join(data_dir, 'train_set_with_wyckoff.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val_set_with_wyckoff.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_set_with_wyckoff.csv'))
    
    # Define original features (first 50)
    original_features = [
        'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'volume', 'n_atoms', 'density',
        'vol_per_atom', 'frac_O', 'frac_F', 'frac_N', 'frac_S', 'frac_Cl', 'frac_Se',
        'frac_Br', 'frac_I', 'frac_H', 'frac_C', 'frac_P', 'frac_B', 'frac_Te',
        'frac_Si', 'frac_As', 'frac_Ge', 'frac_Sn', 'frac_Pb', 'frac_Bi', 'frac_Sb',
        'frac_Li', 'frac_Na', 'frac_K', 'frac_Rb', 'frac_Cs', 'frac_Be', 'frac_Mg',
        'frac_Ca', 'frac_Sr', 'frac_Ba', 'frac_Al', 'frac_Ga', 'frac_In', 'frac_Tl',
        'frac_Sc', 'frac_Y', 'frac_La', 'frac_Ti', 'frac_Zr', 'frac_V'
    ]
    
    # All other features (except target and mp_id) are Wyckoff features
    all_features = [col for col in train_df.columns if col not in ['mp_id', 'ln_klat', 'klat']]
    wyckoff_features = [col for col in all_features if col not in original_features]
    
    print(f"Original features: {len(original_features)}")
    print(f"Wyckoff features: {len(wyckoff_features)}")
    
    # Extract data
    def extract_data(df):
        X_orig = df[original_features].values.astype(np.float32)
        X_wyck = df[wyckoff_features].values.astype(np.float32)
        y = df['ln_klat'].values.astype(np.float32)
        return X_orig, X_wyck, y
    
    return extract_data(train_df), extract_data(val_df), extract_data(test_df)

def train_attention_hybrid(train_data, val_data, device, max_epochs=200):
    """Train the attention hybrid model."""
    
    X_orig_train, X_wyck_train, y_train = train_data
    X_orig_val, X_wyck_val, y_val = val_data
    
    # Scale features
    orig_scaler = StandardScaler()
    wyck_scaler = StandardScaler()
    
    X_orig_train = orig_scaler.fit_transform(X_orig_train)
    X_wyck_train = wyck_scaler.fit_transform(X_wyck_train)
    X_orig_val = orig_scaler.transform(X_orig_val)
    X_wyck_val = wyck_scaler.transform(X_wyck_val)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_orig_train),
        torch.FloatTensor(X_wyck_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_orig_val),
        torch.FloatTensor(X_wyck_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = AttentionHybridBNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.7)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 0
    
    print(f"🚀 Training Attention Hybrid BNN...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_orig, batch_wyck, batch_y in train_loader:
            batch_orig, batch_wyck, batch_y = batch_orig.to(device), batch_wyck.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_orig, batch_wyck)
            loss = F.mse_loss(pred, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_orig, batch_wyck, batch_y in val_loader:
                batch_orig, batch_wyck, batch_y = batch_orig.to(device), batch_wyck.to(device), batch_y.to(device)
                pred = model(batch_orig, batch_wyck)
                loss = F.mse_loss(pred, batch_y)
                val_loss += loss.item()
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_r2 = r2_score(val_targets, val_preds)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Train={train_loss:.4f}, Val={val_loss:.4f} (R²={val_r2:.4f})")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            
        if patience >= 40:
            print(f"Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(best_model_state)
    return model, orig_scaler, wyck_scaler

def evaluate_model(model, test_data, orig_scaler, wyck_scaler, device):
    """Evaluate the model."""
    
    X_orig_test, X_wyck_test, y_test = test_data
    
    # Scale features
    X_orig_test = orig_scaler.transform(X_orig_test)
    X_wyck_test = wyck_scaler.transform(X_wyck_test)
    
    # Create data loader
    test_dataset = TensorDataset(
        torch.FloatTensor(X_orig_test),
        torch.FloatTensor(X_wyck_test),
        torch.FloatTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Evaluation
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_orig, batch_wyck, batch_y in test_loader:
            batch_orig, batch_wyck, batch_y = batch_orig.to(device), batch_wyck.to(device), batch_y.to(device)
            pred = model(batch_orig, batch_wyck)
            predictions.extend(pred.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return {'mae': mae, 'r2': r2, 'predictions': predictions, 'targets': targets}

def run_attention_hybrid_split(split_name):
    """Run attention hybrid BNN on a specific split."""
    
    print(f"\n{'='*60}")
    print(f"🔮 ATTENTION HYBRID BNN: {split_name.upper()}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    # Load data
    data_dir = f"data/{split_name}"
    (X_orig_train, X_wyck_train, y_train), (X_orig_val, X_wyck_val, y_val), (X_orig_test, X_wyck_test, y_test) = load_hybrid_data(data_dir)
    
    print(f"📂 Data loaded:")
    print(f"   Train: {len(y_train)} samples")
    print(f"   Val: {len(y_val)} samples") 
    print(f"   Test: {len(y_test)} samples")
    
    # Check sparsity
    wyck_sparsity = (X_wyck_train == 0).mean()
    print(f"   Wyckoff sparsity: {wyck_sparsity:.1%}")
    
    # Train model
    model, orig_scaler, wyck_scaler = train_attention_hybrid(
        (X_orig_train, X_wyck_train, y_train),
        (X_orig_val, X_wyck_val, y_val),
        device
    )
    
    # Evaluate model
    results = evaluate_model(model, (X_orig_test, X_wyck_test, y_test), orig_scaler, wyck_scaler, device)
    
    print(f"\n📊 ATTENTION HYBRID BNN RESULTS FOR {split_name.upper()}:")
    print(f"Features: 50 original + 228 Wyckoff (attention fusion)")
    print(f"Test R²: {results['r2']:.4f}")
    print(f"Test MAE: {results['mae']:.4f}")
    
    return {
        'split': split_name,
        'test_r2': results['r2'],
        'test_mae': results['mae']
    }

def main():
    """Run Attention Hybrid BNN on all splits."""
    
    print("🔮 ATTENTION HYBRID BNN - ALL SPLITS")
    print("=" * 60)
    print("Innovation:")
    print("- Attention-based fusion of original + Wyckoff features")
    print("- Sparse feature handling through attention")
    print("- Cross-attention between dense and sparse features")
    print()
    
    splits = ['random_split', 'ood_split', 'space_group_split']
    all_results = []
    
    for split in splits:
        try:
            result = run_attention_hybrid_split(split)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Error with {split}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary
    if all_results:
        print(f"\n{'='*60}")
        print("📋 ATTENTION HYBRID BNN RESULTS SUMMARY")
        print(f"{'='*60}")
        
        results_df = pd.DataFrame(all_results)
        
        print("Test Set Performance:")
        print(f"{'Split':<18} {'R²':<8} {'MAE':<8} {'Status'}")
        print("-" * 40)
        
        for _, row in results_df.iterrows():
            r2 = row['test_r2']
            mae = row['test_mae']
            status = "✅ Excellent" if r2 > 0.7 else "🔥 Good" if r2 > 0.5 else "⚠️ Moderate" if r2 > 0.2 else "❌ Poor"
            print(f"{row['split']:<18} {r2:<8.3f} {mae:<8.3f} {status}")
        
        # Save results
        results_df.to_csv('attention_hybrid_bnn_results.csv', index=False)
        
        # Performance analysis
        avg_r2 = results_df['test_r2'].mean()
        positive_r2_count = sum(1 for r2 in results_df['test_r2'] if r2 > 0)
        
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        print(f"   Average R²: {avg_r2:.3f}")
        print(f"   Positive R² splits: {positive_r2_count}/3")
        
        if positive_r2_count >= 2:
            print("🎉 Attention Hybrid BNN shows promising results!")
        else:
            print("⚠️ Some splits still challenging")
        
        print(f"\n✅ Results saved: attention_hybrid_bnn_results.csv")

if __name__ == "__main__":
    main()
