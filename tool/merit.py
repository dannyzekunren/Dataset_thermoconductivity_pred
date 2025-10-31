from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import argparse

def kappa_wlmae(y_true, y_pred, p=2, log_base='e', eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # weights: full weight <=2; decay as (2/y)^p above 2
    w = np.minimum(1.0, (2.0 / np.maximum(y_true, eps))**p)
    # log error (natural or base-10)
    if log_base == 'e':
        err = np.abs(np.log(np.maximum(y_pred, eps)) - np.log(np.maximum(y_true, eps)))
    elif log_base == '10':
        err = np.abs(np.log10(np.maximum(y_pred, eps)) - np.log10(np.maximum(y_true, eps)))
    else:
        raise ValueError("log_base must be 'e' or '10'")
    return np.sum(w * err) / np.sum(w)

def compute_merit(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    kappa_wlmae_natural = kappa_wlmae(y_true, y_pred, log_base='e')
    kappa_wlmae_base10 = kappa_wlmae(y_true, y_pred, log_base='10')
    return {
        'MAE': mae,
        'R2': r2,
        'Kappa_WLMAE_NaturalLog': kappa_wlmae_natural,
        'Kappa_WLMAE_Base10': kappa_wlmae_base10
    }

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--file_dir', type=str, required=True, help='Directory containing the dataset files')
    args = arg_parser.parse_args()

    merits_file = pd.read_csv(f"{args.file_dir}")

    merits = compute_merit(merits_file['target'], merits_file['prediction'])
    for key, value in merits.items():
        print(f"{key}: {value:.6f}")