#!/usr/bin/env python3
"""Train and evaluate linear-regression baselines on the shared representation."""

import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "dataset" / "featurized_data"
MODEL_DIR = PROJECT_ROOT / "models" / "linear_regression"
SPLITS = ("random", "spg", "ood")


def load_dataset(path: Path) -> dict:
    """Load CUDA-backed pickle tensors onto the CPU as well as ordinary pickles."""
    original_torch_load = torch.load

    def cpu_torch_load(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        return original_torch_load(*args, **kwargs)

    torch.load = cpu_torch_load
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    finally:
        torch.load = original_torch_load


def as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def train_split(split: str, save_model: bool = True) -> dict:
    data_path = DATA_DIR / f"{split}_scaled_full_soap_bin_matrix_sym_logk.pkl"
    dataset = load_dataset(data_path)
    x_train = as_numpy(dataset["train_input"])
    y_train = as_numpy(dataset["train_label"]).reshape(-1)
    x_test = as_numpy(dataset["test_input"])
    y_test = as_numpy(dataset["test_label"]).reshape(-1)

    model = LinearRegression(n_jobs=-1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / f"linear_regression_{split}.pkl")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("all", *SPLITS),
        default="all",
        help="Data split to evaluate (default: all).",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save fitted models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = SPLITS if args.split == "all" else (args.split,)
    results = {}

    for split in splits:
        results[split] = train_split(split, save_model=not args.no_save)
        metrics = results[split]
        print(
            f"{split.upper()}: R²={metrics['r2']:.3f}, "
            f"MAE={metrics['mae']:.3f}, MSE={metrics['mse']:.3f}"
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
