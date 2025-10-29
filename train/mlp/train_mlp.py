#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-10-28
Description: Training MLP with attention module for regression tasks
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import random
import wandb
import pandas as pd
import itertools
from copy import deepcopy
from itertools import product
import joblib


class SimpleRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128,64], bottleneck_dim=32):
        super().__init__()
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)
        self.bottleneck = nn.Linear(dims[-1], bottleneck_dim)
        self.head = nn.Linear(bottleneck_dim, 1)

    def forward(self, x):
        h = self.encoder(x)
        z = torch.relu(self.bottleneck(h))
        y = self.head(z)
        return y


class AttentionRegressor(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(input_dim, embed_dim)

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: [batch, feature_dim]
        """
        x_proj = self.input_proj(x)  # [batch, embed_dim]

        x_seq = x_proj.unsqueeze(1)  # [batch, 1, embed_dim]

        attn_out, _ = self.mha(x_seq, x_seq, x_seq)  # [batch, 1, embed_dim]
        x = self.layernorm1(x_seq + attn_out)

        ffn_out = self.ffn(x)
        x = self.layernorm2(x + ffn_out)

        x = x.squeeze(1)

        out = self.output_head(x)
        return out


class MultiFeatureAttentionRegressor(nn.Module):
    def __init__(self,
                 comp_dim, soap_dim, symm_dim,
                 embed_dim=128, num_heads=4, num_layers=2,
                 dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim

        self.comp_proj = nn.Sequential(
            nn.Linear(comp_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        self.soap_proj = nn.Sequential(
            nn.Linear(soap_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        self.symm_proj = nn.Sequential(
            nn.Linear(symm_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x_comp, x_soap, x_symm):
        """
        x_comp: [batch, comp_dim]
        x_soap: [batch, soap_dim]
        x_symm: [batch, symm_dim]
        """
        t1 = self.comp_proj(x_comp)
        t2 = self.soap_proj(x_soap)
        t3 = self.symm_proj(x_symm)

        tokens = torch.stack([t1, t2, t3], dim=1)

        encoded = self.encoder(tokens)  # [batch, 3, embed_dim]

        pooled = encoded.mean(dim=1)

        out = self.head(pooled)
        return out

def train_simple_model(X_train, y_train,
                       hidden_dims=[256,128],
                       bottleneck_dim=16,
                       lr=1e-3,
                       epochs=200,
                       batch_size=32,
                       val_split: None | float = 0.1,
                       X_val=None,
                       y_val=None,
                       wandb_log=False,
                       project_name="Auto-KANppa",
                       run_name=None,
                       device=None,
                       seed=42):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(device)

    # train-validation split
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32).view(-1,1))
    if val_split is None:
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val must be provided when val_split is None.")
        train_ds = dataset
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.float32).view(-1,1))
    else:
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # model, optimizer, loss
    model = SimpleRegressor(X_train.shape[1], hidden_dims=hidden_dims, bottleneck_dim=bottleneck_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # wandb
    if wandb_log:
        wandb.init(project=project_name, name=run_name)
        wandb.config.update({
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "hidden_dims": hidden_dims, "bottleneck_dim": bottleneck_dim, "val_split": val_split
        })

    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        # validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                y_pred = model(xb)
                val_preds.append(y_pred.cpu().numpy())
                val_targets.append(yb.cpu().numpy())
        val_preds = np.concatenate(val_preds, axis=0).reshape(-1)
        val_targets = np.concatenate(val_targets, axis=0).reshape(-1)
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets, val_preds)

        if wandb_log:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_mae": val_mae, "val_r2": val_r2})

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} | Val R2: {val_r2:.4f}")

    if wandb_log:
        wandb.finish()

    return model, val_mae, val_r2

def train_attention_model(X_train, y_train, val_split: None|float=0.1, X_val=None, y_val=None,
                          embed_dim=256, num_heads=8,
                          lr=0.0005, epochs=200, batch_size=64,
                          project="Auto-KANppa",
                          run_name="attention", use_wandb=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionRegressor(X_train.shape[1], embed_dim, num_heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if use_wandb:
        import wandb
        wandb.init(project=project, name=run_name, config={
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "embed_dim": embed_dim, "num_heads": num_heads
        })

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    if val_split is not None:
        n_val = int(len(X_train) * val_split)
        n_train = len(X_train) - n_val
        X_train_t, X_val_t = torch.split(X_train_t, [n_train, n_val])
        y_train_t, y_val_t = torch.split(y_train_t, [n_train, n_val])
    else:
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val must be provided when val_split is None.")
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i+batch_size]
            Xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            y_pred = model(Xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val_t)
            val_loss = criterion(y_pred_val, y_val_t).item()
            rmse = torch.sqrt(torch.mean((y_pred_val - y_val_t)**2)).item()

        if use_wandb:
            wandb.log({"train_loss": loss.item(), "val_loss": val_loss, "val_RMSE": rmse})

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_RMSE={rmse:.4f}")

    if use_wandb:
        wandb.finish()

    return model


def train_multifeature_attention(
        Xc_train, Xs_train, Xsym_train, y_train,
        Xc_val, Xs_val, Xsym_val, y_val,
        comp_dim, soap_dim, symm_dim,
        embed_dim=128, num_heads=4, num_layers=2,
        lr=1e-3, epochs=300, batch_size=64,
        dropout=0.1, project="thermal_multi_attention", use_wandb=True):
    import wandb
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiFeatureAttentionRegressor(comp_dim, soap_dim, symm_dim,
                                           embed_dim, num_heads, num_layers, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if use_wandb:
        wandb.init(project=project, config={
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "embed_dim": embed_dim, "num_heads": num_heads, "num_layers": num_layers
        })

    # 转换数据
    Xc_train_t = torch.tensor(Xc_train, dtype=torch.float32).to(device)
    Xs_train_t = torch.tensor(Xs_train, dtype=torch.float32).to(device)
    Xsym_train_t = torch.tensor(Xsym_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    Xc_val_t = torch.tensor(Xc_val, dtype=torch.float32).to(device)
    Xs_val_t = torch.tensor(Xs_val, dtype=torch.float32).to(device)
    Xsym_val_t = torch.tensor(Xsym_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xc_train_t))
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb_c, xb_s, xb_sym = Xc_train_t[idx], Xs_train_t[idx], Xsym_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            y_pred = model(xb_c, xb_s, xb_sym)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            y_pred_val = model(Xc_val_t, Xs_val_t, Xsym_val_t)
            val_loss = criterion(y_pred_val, y_val_t).item()
            rmse = torch.sqrt(torch.mean((y_pred_val - y_val_t) ** 2)).item()

        if use_wandb:
            wandb.log({"train_loss": loss.item(), "val_loss": val_loss, "val_RMSE": rmse})

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_RMSE={rmse:.4f}")

    if use_wandb:
        wandb.finish()

    return model


def predict(model, X, log_transform=True, device="cuda" if torch.cuda.is_available() else "cpu"):
    device = torch.device(device)
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_t).cpu().numpy().flatten()
    if log_transform:
        return np.exp(y_pred)
    return y_pred

def grid_search(X_train, y_train,
                             param_grid,
                             epochs=200,
                             batch_size=64,
                             val_split=0.1,
                             device=None,
                             seed=42,
                             wandb_log=False,
                             project_name="thermal_baseline"):

    """
    param_grid: dict, key->list of values
        e.g., {"hidden_dims": [[128,64],[256,128]], "bottleneck_dim":[16,32], "lr":[1e-3,5e-4]}
    Returns:
        best_model, best_params, all_results (list of dict)
    """
    all_params = list(product(*param_grid.values()))
    keys = list(param_grid.keys())

    best_mae = float('inf')
    best_model = None
    best_params = None
    all_results = []

    for idx, vals in enumerate(all_params):
        params = dict(zip(keys, vals))
        run_name = f"grid_{idx+1}"
        print(f"\n=== Grid {idx+1}/{len(all_params)} | Params: {params} ===")

        model, val_mae, val_r2 = train_simple_model(
            X_train, y_train,
            hidden_dims=params.get("hidden_dims",[128,64]),
            bottleneck_dim=params.get("bottleneck_dim",32),
            lr=params.get("lr",1e-3),
            epochs=epochs,
            batch_size=batch_size,
            val_split=val_split,
            wandb_log=wandb_log,
            project_name=project_name,
            run_name=run_name,
            device=device,
            seed=seed
        )

        all_results.append({**params, "val_mae": val_mae, "val_r2": val_r2})

        if val_mae < best_mae:
            best_mae = val_mae
            best_model = model
            best_params = params

    print("\n=== Grid Search Complete ===")
    print(f"Best Params: {best_params} | Val MAE: {best_mae:.4f} | Val R2: {all_results[[r['val_mae'] for r in all_results].index(best_mae)]['val_r2']:.4f}")

    return best_model, best_params, all_results

def grid_search_attention_model(
    X_train, y_train,
    X_val=None, y_val=None,
    val_split=0.1,
    param_grid=None,
    project="Auto-KANppa",
    use_wandb=False,
    device=None
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    keys, values = zip(*param_grid.items())
    all_combinations = list(itertools.product(*values))

    results = []

    for idx, combo in enumerate(all_combinations):
        params = dict(zip(keys, combo))
        print(f"\n===== Running combination {idx+1}/{len(all_combinations)} =====")
        print(params)

        model = train_attention_model(
            X_train, y_train,
            val_split=val_split,
            X_val=X_val,
            y_val=y_val,
            embed_dim=params.get("embed_dim", 128),
            num_heads=params.get("num_heads", 4),
            lr=params.get("lr", 1e-3),
            epochs=params.get("epochs", 200),
            batch_size=params.get("batch_size", 64),
            project=project,
            run_name=f"grid_{idx+1}",
            use_wandb=use_wandb
        )

        if X_val is None or y_val is None:
            n_val = int(len(X_train) * val_split)
            X_train_split = X_train[:-n_val]
            X_val_split = X_train[-n_val:]
            y_train_split = y_train[:-n_val]
            y_val_split = y_train[-n_val:]
        else:
            X_val_split, y_val_split = X_val, y_val

        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val_split, dtype=torch.float32).to(device)
            y_val_t = torch.tensor(y_val_split, dtype=torch.float32).view(-1, 1).to(device)
            y_pred = model(X_val_t)
            rmse = torch.sqrt(torch.mean((y_pred - y_val_t) ** 2)).item()

        print(f"Validation RMSE = {rmse:.5f}")
        result_entry = deepcopy(params)
        result_entry["val_RMSE"] = rmse
        results.append(result_entry)

    results_df = pd.DataFrame(results)
    best_idx = results_df["val_RMSE"].idxmin()
    best_params = results_df.loc[best_idx].to_dict()

    print("\n===== Best configuration =====")
    print(best_params)

    return results_df, best_params


if __name__ == "__main__":
    import pickle

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open("dataset/featurized_data/random_scaled_full_soap_bin_matrix_sym_logk.pkl", "rb") as f:
        dataset = pickle.load(f)

    X_train = dataset['train_input']
    y_train = dataset['train_label']
    X_test = dataset['test_input']
    y_test = dataset['test_label']

    model, _, _ = train_simple_model(X_train, y_train, hidden_dims=[256,128],
                               bottleneck_dim=16,
                               lr=1e-3,
                               epochs=200,
                               batch_size=32,
                               val_split=None,
                               X_val=X_test,
                               y_val=y_test,
                               wandb_log=True,
                               project_name="Auto-KANppa",
                               run_name="mlp-ood",
                               device=None,
                               seed=42)

    # model = train_attention_model(X_train, y_train, val_split=None, X_val=X_test, y_val=y_test,
    #                               embed_dim=256, num_heads=8,
    #                               lr=0.0005, epochs=200, batch_size=64,
    #                               project="Auto-KANppa",
    #                               run_name="attention", use_wandb=True)

    y_pred_test = predict(model, X_test, log_transform=False)
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.detach().cpu().numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.detach().cpu().numpy()

    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    print("Final Test MAE (log y):", mae)
    print("Final Test R² (log y):", r2)

    joblib.dump(model, "mlp_ood.pkl")
