#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-xx-xx
Description: [Brief description of the script's purpose]
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-10-09
Description: [Brief description of the script's purpose]
"""
#%%time
import pickle
import numpy as np
import wandb
from varkan.effkan import KAN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import tempfile


#%% Load featurized data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(
    project="auto-KANppa",
    name="eff-kan-ood",
    config={
        "model_type": "effKAN",
        "hidden_dims": 32,
        "grid": 5,
        "k": 3,
        "optimizer": "Adam",
        "lr": 1e-3,
        "lamb": 1e-2,
        "steps": 250,
        "seed": 7,
        "device": str(device),
    }
)

with open("dataset/featurized_data/ood_scaled_full_soap_bin_matrix_sym_logk.pkl", "rb") as f:
    dataset = pickle.load(f)

X_train = dataset['train_input']
y_train = dataset['train_label']
X_test = dataset['test_input']
y_test = dataset['test_label']

config = wandb.config

#%% Train
model = KAN([dataset['train_input'].shape[1], config.hidden_dims, 1]).to(device)
# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
# Define loss
criterion = nn.MSELoss()
for step in range(config.steps):
    # Train
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        out = model(X_test)
        val_loss = criterion(out, y_test).item()
        print(f"Epoch [{step+1}/{config.steps}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")


y_pred = model(X_test)
if isinstance(X_test, torch.Tensor):
    X_test = X_test.detach().cpu().numpy()
if isinstance(y_test, torch.Tensor):
    y_test = y_test.detach().cpu().numpy()
if isinstance(y_pred, torch.Tensor):
    y_pred = y_pred.detach().cpu().numpy()

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Test R2:  {r2:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test MSE: {mse:.4f}")

wandb.log({
    "R2": r2,
    "MAE": mae,
    "MSE": mse
})

joblib.dump(model, "kan_ood.pkl")
wandb.save("kan_ood.pkl")

wandb.finish()
