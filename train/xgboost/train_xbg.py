#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-10-29
Description: Train XGBoost model for thermal conductivity prediction
"""
import pickle
import numpy as np
import torch
import wandb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt
import tempfile
import joblib


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("dataset/featurized_data/ood_scaled_full_soap_bin_matrix_sym_logk.pkl", "rb") as f:
    dataset = pickle.load(f)

X_train = dataset['train_input']
y_train = dataset['train_label']
print(X_train.shape)
X_test = dataset['test_input']
y_test = dataset['test_label']


# Hyperparameter grid search (commented out after finding best params)
# param_grid = {
#     "n_estimators": [500, 1000],
#     "max_depth": [4, 6, 8],
#     "learning_rate": [0.01],
#     "subsample": [0.8],
#     "colsample_bytree": [1.0],
#     "min_child_weight": [1, 3, 5],
#     "reg_lambda": [0.1],
#     "reg_alpha": [0.1],
# }
#
# keys, values = zip(*param_grid.items())
# param_combinations = [dict(zip(keys, v)) for v in product(*values)]
#
# best_params = None
# best_score = np.inf
#
# print(f"🔍 Total combinations to test: {len(param_combinations)}")
#
# for i, params in enumerate(param_combinations):
#     start = time.time()
#     model = XGBRegressor(
#         **params,
#         random_state=42,
#         n_jobs=-1,
#         verbosity=0,
#         early_stopping_rounds=20,
#     )
#     model.fit(
#         X_train,
#         y_train,
#         eval_set=[(X_test, y_test)],
#         verbose=False,
#     )
#
#     y_pred = model.predict(X_test)
#
#     if isinstance(X_test, torch.Tensor):
#         X_test = X_test.detach().cpu().numpy()
#     if isinstance(y_test, torch.Tensor):
#         y_test = y_test.detach().cpu().numpy()
#
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#
#     print(
#         f"[{i+1}/{len(param_combinations)}] "
#         f"MAE={mae:.4f}, MSE={mse:.4f}, params={params}, "
#         f"time={time.time()-start:.1f}s"
#     )
#
#     if mse < best_score:
#         best_score = mse
#         best_params = params
#
# print("\n✅ Best hyperparameters found:")
# for k, v in best_params.items():
#     print(f"  {k}: {v}")
# print(f"Best MSE: {best_score:.4f}")

best_params = {
    "n_estimators": 1000,
    "max_depth": 8,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "reg_lambda": 0.1,
    "reg_alpha": 0.1
}

wandb.init(
    project="Auto-KANppa",
    name="xgb-ood",
    config=best_params
)
config = wandb.config

final_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0, early_stopping_rounds=20)
final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

y_pred = final_model.predict(X_test)

if isinstance(X_test, torch.Tensor):
    X_test = X_test.detach().cpu().numpy()
if isinstance(y_test, torch.Tensor):
    y_test = y_test.detach().cpu().numpy()

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² = {r2:.3f}, MAE = {mae:.3f}, MSE = {mse:.3f}")
wandb.log({
    "R2": r2,
    "MAE": mae,
    "MSE": mse
})

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("True Thermal Conductivity (log K)")
plt.ylabel("Predicted Thermal Conductivity (log K)")
plt.title(f"XGBoost OOD (R²={r2:.3f})")

with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    plt.savefig(tmp.name)
    wandb.log({"Prediction Scatter": wandb.Image(tmp.name)})
plt.close()

joblib.dump(final_model, "xgb_ood.pkl")
wandb.save("xgb_ood.pkl")

wandb.finish()
