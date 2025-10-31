#!/usr/bin/env python
"""Simple prediction script that directly loads .pt model and predicts."""

import os
import sys
import csv
import json
import torch
import argparse
import numpy as np
from typing import Dict, Any
from tqdm import tqdm

try:
    from jarvis.core.atoms import Atoms
    from jarvis.core.graphs import Graph
    from aliegnn.models.alignn_egnn import ALIEGNN
    from aliegnn.config import TrainingConfig
    from jarvis.db.jsonutils import loadjson
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install: pip install jarvis-tools")
    sys.exit(1)


def load_model(model_path: str, config_path: str) -> tuple:
    """
    Load model from .pt file.

    Args:
        model_path: Path to .pt model file
        config_path: Path to config.json file

    Returns:
        (model, config, device)
    """
    print(f"Loading config from: {config_path}")
    config_dict = loadjson(config_path)
    config = TrainingConfig(**config_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Create model
    model = ALIEGNN(config.model)

    # Load weights - handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print("✓ Model loaded successfully")
    return model, config, device


def predict_structure(
    model: torch.nn.Module,
    atoms: Atoms,
    config: TrainingConfig,
    device: torch.device
) -> float:
    """
    Predict property for a single structure.

    Args:
        model: Loaded ALIEGNN model
        atoms: Jarvis Atoms object
        config: Training configuration
        device: torch device

    Returns:
        Predicted value
    """
    # Create graph
    g, lg = Graph.atom_dgl_multigraph(
        atoms=atoms,
        neighbor_strategy=config.neighbor_strategy,
        cutoff=float(config.cutoff),
        max_neighbors=config.max_neighbors,
        atom_features=config.atom_features,
        compute_line_graph=True,
        use_canonize=config.use_canonize
    )

    # Move to device
    g = g.to(device)
    lg = lg.to(device)

    # Predict
    with torch.no_grad():
        prediction = model([g, lg])
        result = prediction.detach().cpu().numpy().flatten()

    return float(result[0])


def load_structure(file_path: str, file_format: str = "auto") -> Atoms:
    """
    Load structure from file.

    Args:
        file_path: Path to structure file
        file_format: File format (auto/poscar/cif/xyz)

    Returns:
        Jarvis Atoms object
    """
    if file_format == "auto":
        # Auto-detect from extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.cif':
            return Atoms.from_cif(file_path)
        elif ext.lower() == '.xyz':
            return Atoms.from_xyz(file_path, box_size=500)
        elif 'POSCAR' in file_path or 'CONTCAR' in file_path or ext.lower() == '.vasp':
            return Atoms.from_poscar(file_path)
        else:
            # Default to poscar
            return Atoms.from_poscar(file_path)
    elif file_format == "poscar":
        return Atoms.from_poscar(file_path)
    elif file_format == "cif":
        return Atoms.from_cif(file_path)
    elif file_format == "xyz":
        return Atoms.from_xyz(file_path, box_size=500)
    else:
        raise ValueError(f"Unknown file format: {file_format}")


def predict_single(
    model: torch.nn.Module,
    config: TrainingConfig,
    device: torch.device,
    structure_file: str
) -> float:
    """Predict for a single structure and print result."""
    atoms = load_structure(structure_file)
    prediction = predict_structure(model, atoms, config, device)

    formula = atoms.composition.reduced_formula if hasattr(atoms, 'composition') else "Unknown"

    print(f"\nPrediction for {os.path.basename(structure_file)}:")
    print(f"  Formula: {formula}")
    print(f"  Predicted value: {prediction:.6f}")

    return prediction


def predict_from_csv(
    model: torch.nn.Module,
    config: TrainingConfig,
    device: torch.device,
    csv_file: str,
    structures_dir: str = None,
    output_dir: str = "predictions"
) -> None:
    """
    Predict for structures listed in CSV file.

    CSV format: mp-id,(or target_value (first column is mp-id))
    Structures are in structures_dir with folder name matching mp-id.

    Args:
        model: Loaded model
        config: Configuration
        device: torch device
        csv_file: CSV file with mp-id in first column
        structures_dir: Directory with structure folders (default: same as CSV dir)
        output_dir: Output directory for predictions.csv and predictions.json
    """
    # Default structures_dir to same directory as CSV
    if structures_dir is None:
        structures_dir = os.path.dirname(csv_file) or "."

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "predictions.csv")
    output_json = os.path.join(output_dir, "predictions.json")
    output_summary = os.path.join(output_dir, "SUMMARY.txt")

    print(f"Reading CSV: {csv_file}")
    print(f"Looking for structures in: {structures_dir}")
    print(f"Output directory: {output_dir}\n")

    # Read CSV
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and len(row) >= 1:
                mp_id = row[0].strip()
                ground_truth = float(row[1]) if len(row) >= 2 else None
                data.append((mp_id, ground_truth))

    print(f"Found {len(data)} materials in CSV\n")

    results = []
    errors = []

    for i, (mp_id, ground_truth) in (enumerate(tqdm(data), 1)):
        # Find structure - look in folder named mp_id
        structure_path = os.path.join(structures_dir, mp_id)

        if not os.path.exists(structure_path):
            print(f"[{i}/{len(data)}] ✗ {mp_id}: Not found")
            errors.append(mp_id)
            results.append({
                "mp_id": mp_id,
                "ground_truth": ground_truth,
                "prediction": None,
                "error": "Structure not found"
            })
            continue

        try:
            # Load structure (handle directory or file)
            if os.path.isdir(structure_path):
                # Look for POSCAR/CONTCAR in directory
                poscar_path = None
                for filename in ['POSCAR', 'CONTCAR']:
                    path = os.path.join(structure_path, filename)
                    if os.path.exists(path):
                        poscar_path = path
                        break

                if poscar_path is None:
                    raise FileNotFoundError(f"No POSCAR/CONTCAR in {structure_path}")

                atoms = Atoms.from_poscar(poscar_path)
            else:
                atoms = load_structure(structure_path)

            # Predict
            prediction = predict_structure(model, atoms, config, device)

            # Calculate errors
            abs_error = None
            rel_error = None
            if ground_truth is not None:
                abs_error = abs(prediction - ground_truth)
                if ground_truth != 0:
                    rel_error = abs_error / abs(ground_truth) * 100

            formula = atoms.composition.reduced_formula if hasattr(atoms, 'composition') else "Unknown"

            result = {
                "mp_id": mp_id,
                "formula": formula,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "absolute_error": abs_error,
                "relative_error_percent": rel_error
            }

            results.append(result)

        except Exception as e:
            print(f"[{i}/{len(data)}] ✗ {mp_id}: {e}")
            errors.append(mp_id)
            results.append({
                "mp_id": mp_id,
                "ground_truth": ground_truth,
                "prediction": None,
                "error": str(e)
            })

    # Save results
    with open(output_csv, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    # Calculate summary statistics
    successful = [r for r in results if r.get('prediction') is not None]
    summary_lines = []

    summary_lines.append("="*70)
    summary_lines.append("PREDICTION SUMMARY")
    summary_lines.append("="*70)
    summary_lines.append(f"CSV File:         {csv_file}")
    summary_lines.append(f"Structures Dir:   {structures_dir}")
    summary_lines.append(f"Output Dir:       {output_dir}")
    summary_lines.append("")
    summary_lines.append(f"Total samples:    {len(data)}")
    summary_lines.append(f"Successful:       {len(successful)}")
    summary_lines.append(f"Failed:           {len(errors)}")

    if successful:
        preds = [r['prediction'] for r in successful]
        summary_lines.append("")
        summary_lines.append("Prediction Statistics:")
        summary_lines.append(f"  Mean:  {np.mean(preds):.4f}")
        summary_lines.append(f"  Std:   {np.std(preds):.4f}")
        summary_lines.append(f"  Min:   {np.min(preds):.4f}")
        summary_lines.append(f"  Max:   {np.max(preds):.4f}")

        # Metrics if ground truth available
        with_gt = [r for r in successful if r.get('ground_truth') is not None]
        if with_gt:
            gt = [r['ground_truth'] for r in with_gt]
            pred = [r['prediction'] for r in with_gt]
            abs_errors = [abs(p - g) for p, g in zip(pred, gt)]

            mae = np.mean(abs_errors)

            ss_res = sum((p - g)**2 for p, g in zip(pred, gt))
            ss_tot = sum((g - np.mean(gt))**2 for g in gt)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            summary_lines.append("")
            summary_lines.append("Error Metrics (vs Ground Truth):")
            summary_lines.append(f"  MAE:        {mae:.4f}")
            summary_lines.append(f"  R²:         {r2:.4f}")

            # Calculate MAPE
            mapes = [r['relative_error_percent'] for r in with_gt if r.get('relative_error_percent') is not None]
            if mapes:
                summary_lines.append(f"  MAPE:       {np.mean(mapes):.2f}%")

    if errors:
        summary_lines.append("")
        summary_lines.append(f"Failed materials ({len(errors)}):")
        for err_id in errors[:10]:
            summary_lines.append(f"  - {err_id}")
        if len(errors) > 10:
            summary_lines.append(f"  ... and {len(errors) - 10} more")

    summary_lines.append("")
    summary_lines.append("="*70)

    # Print to console
    print("\n" + "\n".join(summary_lines))

    # Save summary to file
    with open(output_summary, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))

    print(f"\n✓ Results saved to:")
    print(f"  - CSV:     {output_csv}")
    print(f"  - JSON:    {output_json}")
    print(f"  - SUMMARY: {output_summary}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Simple ALIEGNN prediction - directly load .pt and predict"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to .pt model file (e.g., best_model.pt)"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.json file"
    )

    parser.add_argument(
        "--structure",
        default=None,
        help="Single structure file to predict (POSCAR/cif/xyz)"
    )

    parser.add_argument(
        "--csv",
        default=None,
        help="CSV file with mp-id in first column (format: mp-id,target_value)"
    )

    parser.add_argument(
        "--structures_dir",
        default=None,
        help="Directory containing structure files (default: same dir as CSV)"
    )

    parser.add_argument(
        "--output",
        default="predictions",
        help="Output directory (will contain predictions.csv and predictions.json)"
    )

    parser.add_argument(
        "--format",
        default="auto",
        choices=["auto", "poscar", "cif", "xyz"],
        help="Structure file format"
    )

    args = parser.parse_args()

    # Load model
    model, config, device = load_model(args.model, args.config)

    # Predict
    if args.structure:
        # Single structure prediction
        predict_single(model, config, device, args.structure)

    elif args.csv:
        # Batch prediction from CSV with mp-id
        structures_dir = args.structures_dir if args.structures_dir else os.path.dirname(args.csv)
        predict_from_csv(
            model, config, device,
            args.csv, structures_dir, args.output
        )

    else:
        print("Error: Please provide --structure, --csv")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
