# -*- coding: utf-8 -*-
"""
Utility to prepare thermal conductivity targets for model assessment.
Splits include a disjoint space-group split and an out-of-distribution split based on
thermal conductivity quantiles. Visualisations are saved alongside the split artifacts.
"""

import json
import pickle
from pathlib import Path
from typing import List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def get_symmetry_metadata(structure: Structure) -> Tuple[Structure, str, int]:
    """Return the symmetrized structure plus space-group symbol and number."""
    sga = SpacegroupAnalyzer(structure, symprec=1e-3)
    refined = sga.get_refined_structure()
    sga = SpacegroupAnalyzer(refined, symprec=0.01)
    symmetrized = sga.get_symmetrized_structure()
    return symmetrized, sga.get_space_group_symbol(), sga.get_space_group_number()


def symm_structure(structure: Structure) -> Structure:
    """Compatibility wrapper returning only the symmetrized structure."""
    symmetrized, _, _ = get_symmetry_metadata(structure)
    return symmetrized


def compute_symmetry_payload(structures: Sequence[Structure]) -> Tuple[List[Structure], List[str], List[int]]:
    """Compute symmetrized structures and associated space-group metadata."""
    symmetrized_structs: List[Structure] = []
    spacegroup_symbols: List[str] = []
    spacegroup_numbers: List[int] = []
    for structure in structures:
        symmetrized, sg_symbol, sg_number = get_symmetry_metadata(structure)
        symmetrized_structs.append(symmetrized)
        spacegroup_symbols.append(sg_symbol)
        spacegroup_numbers.append(sg_number)
    return symmetrized_structs, spacegroup_symbols, spacegroup_numbers


def build_filtered_dataframe(
    mp_ids: Sequence[str],
    klat: np.ndarray,
    kc: np.ndarray,
    kp: np.ndarray,
    spacegroup_symbols: Sequence[str],
    spacegroup_numbers: Sequence[int],
) -> pd.DataFrame:
    """Create dataframe of valid entries with log-transformed klat."""
    records = []
    for idx, (mp_id, kl_value, kc_value, kp_value, sg_symbol, sg_number) in enumerate(
        zip(mp_ids, klat, kc, kp, spacegroup_symbols, spacegroup_numbers)
    ):
        if not np.isfinite(kl_value) or kl_value <= 0 or kl_value > 1e5:
            continue
        records.append(
            {
                "original_index": idx,
                "mp_id": mp_id,
                "klat": float(kl_value),
                "kc": float(kc_value),
                "kp": float(kp_value),
                "spacegroup_symbol": sg_symbol,
                "spacegroup_number": int(sg_number),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No valid entries remain after filtering klat values.")

    df["log_klat"] = np.log(df["klat"])
    return df


def split_by_space_group(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, Set[str], Set[str]]:
    """Select space groups for the test split so train/test groups are disjoint."""
    group_counts = df.groupby("spacegroup_symbol").size()
    unique_groups = group_counts.index.to_numpy()

    if unique_groups.size < 2:
        raise ValueError("At least two unique space groups are required for disjoint splitting.")

    rng = np.random.default_rng(random_state)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    target_test_samples = max(1, int(round(len(df) * test_fraction)))
    test_groups: List[str] = []
    accumulated = 0
    for group in shuffled_groups:
        test_groups.append(group)
        accumulated += int(group_counts.loc[group])
        if accumulated >= target_test_samples:
            break

    if len(test_groups) == unique_groups.size:
        test_groups.pop()

    test_group_set: Set[str] = set(test_groups)
    train_group_set: Set[str] = set(unique_groups) - test_group_set

    if not test_group_set or not train_group_set:
        # Fallback: move one group across to avoid empty split.
        swap_group = shuffled_groups[0]
        if swap_group in test_group_set:
            test_group_set.remove(swap_group)
            train_group_set.add(swap_group)
        else:
            test_group_set.add(swap_group)
            train_group_set.discard(swap_group)

    train_mask = df["spacegroup_symbol"].isin(train_group_set)
    test_mask = df["spacegroup_symbol"].isin(test_group_set)

    return train_mask, test_mask, train_group_set, test_group_set


def build_random_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series]:
    """Create a standard random 80/20 train/test split with no mp_id overlap."""
    from sklearn.model_selection import train_test_split
    
    # Get unique mp_ids and split them randomly
    unique_mp_ids = df["mp_id"].unique()
    train_mp_ids, test_mp_ids = train_test_split(
        unique_mp_ids,
        test_size=test_fraction,
        random_state=random_state
    )
    
    # Create masks based on mp_id assignment
    train_mask = df["mp_id"].isin(train_mp_ids)
    test_mask = df["mp_id"].isin(test_mp_ids)
    
    return train_mask, test_mask


def build_ood_split(
    df: pd.DataFrame,
    test_threshold: float = 1.0,
    train_threshold: float = 0.8,
) -> Tuple[pd.Series, pd.Series, float, float]:
    """Create an out-of-distribution split with non-overlapping thresholds.
    
    Test set: klat < test_threshold (strictly)
    Train set: klat > train_threshold (strictly)
    
    For duplicate mp_ids with instances spanning thresholds:
    - If ALL instances of an mp_id are < test_threshold, assign to test
    - If ALL instances of an mp_id are > train_threshold, assign to train
    - If instances span the gap or thresholds, exclude entirely to maintain clean splits
    """
    if train_threshold >= test_threshold:
        raise ValueError("Train threshold must be less than test threshold to create a gap.")

    # Group by mp_id and check if all instances meet the criteria
    test_candidates = set()
    train_candidates = set()
    
    for mp_id, group in df.groupby("mp_id"):
        klat_values = group["klat"].values
        
        # Check if all instances are strictly in test range
        if (klat_values < test_threshold).all():
            test_candidates.add(mp_id)
        # Check if all instances are strictly in train range
        elif (klat_values > train_threshold).all():
            train_candidates.add(mp_id)
        # Otherwise, exclude this mp_id (instances span thresholds)
    
    final_test_mask = df["mp_id"].isin(test_candidates)
    final_train_mask = df["mp_id"].isin(train_candidates)

    if final_test_mask.sum() == 0:
        raise ValueError("No samples selected for the out-of-distribution test set.")
    
    if final_train_mask.sum() == 0:
        raise ValueError("No samples selected for the out-of-distribution train set.")

    return final_train_mask, final_test_mask, train_threshold, test_threshold


def plot_space_group_highlight(
    df: pd.DataFrame,
    train_groups: Set[str],
    test_groups: Set[str],
    output_path: Path,
) -> None:
    """Visualise space-group coverage for train/test splits."""
    sorted_df = df.sort_values(["spacegroup_number", "mp_id"]).reset_index(drop=True)
    if sorted_df.empty:
        return

    group_order: List[str] = []
    spans: List[Tuple[int, int]] = []
    left = 0
    for group, group_df in sorted_df.groupby("spacegroup_symbol", sort=False):
        count = len(group_df)
        group_order.append(group)
        spans.append((left, count))
        left += count

    cmap = plt.get_cmap("tab20", len(group_order) if group_order else 1)
    color_lookup = {group: cmap(i % cmap.N) for i, group in enumerate(group_order)}
    muted_colour = "#d9d9d9"

    fig, axes = plt.subplots(2, 1, figsize=(12, 2.6), constrained_layout=True)
    highlight_sets = [
        ("Train space groups", train_groups),
        ("Test space groups", test_groups),
    ]

    total_width = spans[-1][0] + spans[-1][1] if spans else len(sorted_df)

    for ax, (title, highlight_set) in zip(axes, highlight_sets):
        for (start, width), group in zip(spans, group_order):
            colour = color_lookup[group] if group in highlight_set else muted_colour
            ax.broken_barh([(start, width)], (0.1, 0.8), facecolors=colour)
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontsize=11)

    # Label only selected space groups to avoid overlap
    # Select every Nth space group based on total count to get ~15-20 labels
    bottom_ax = axes[-1]
    num_labels_target = min(20, len(group_order))
    label_step = max(1, len(group_order) // num_labels_target)
    
    for idx, ((start, width), group) in enumerate(zip(spans, group_order)):
        # Only label selected space groups (every Nth one, plus largest groups)
        if idx % label_step == 0 or width > 100:
            x_pos = start + width / 2
            bottom_ax.text(
                x_pos,
                -0.15,
                group,
                rotation=45,
                ha="right",
                va="center",
                fontsize=9,
                color=color_lookup[group],
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_klat_histogram(train_values: np.ndarray, test_values: np.ndarray, output_path: Path) -> None:
    """Overlay histograms of log(klat) for train and test splits."""
    if train_values.size == 0 or test_values.size == 0:
        raise ValueError("Train and test values must be non-empty for histogram plotting.")

    # Convert to log scale for better visualization
    train_log = np.log(train_values)
    test_log = np.log(test_values)
    
    combined = np.concatenate([train_log, test_log])
    min_val = combined.min()
    max_val = combined.max()
    if min_val == max_val:
        max_val = min_val + 1e-6

    bins = np.linspace(min_val, max_val, 50)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(train_log, bins=bins, alpha=0.6, label="Train", color="#1f77b4", edgecolor="white")
    ax.hist(test_log, bins=bins, alpha=0.6, label="Test", color="#d62728", edgecolor="white")
    ax.set_xlabel("log(k_lattice)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Out-of-distribution klattice split (log scale)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def dump_split_payload(payload: dict, output_path: Path) -> None:
    """Persist split information via pickle for downstream reuse."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle)


def main() -> None:
    data_path = Path("mp_ids_thermal_conductivity.npz")
    structures_path = Path("Structures_not_sym.json")

    data = np.load(data_path)
    kc = data["kc"]
    kp = data["kp"]
    klat = data["klat"]
    mp_ids = data["mp_ids"]

    with structures_path.open() as handle:
        pymatgen_dict_structures_not_sym = json.load(handle)

    raw_structures = [Structure.from_dict(s) for s in pymatgen_dict_structures_not_sym]
    pymatgen_structures, spacegroup_symbols, spacegroup_numbers = compute_symmetry_payload(raw_structures)

    df = build_filtered_dataframe(
        mp_ids=mp_ids,
        klat=klat,
        kc=kc,
        kp=kp,
        spacegroup_symbols=spacegroup_symbols,
        spacegroup_numbers=spacegroup_numbers,
    )

    print(f"Filtered dataset contains {len(df)} samples (from {len(mp_ids)} original entries).")

    # Standard random 80/20 split
    train_mask_random, test_mask_random = build_random_split(df)
    random_split = {
        "split_type": "random_80_20",
        "train_mp_ids": df.loc[train_mask_random, "mp_id"].tolist(),
        "test_mp_ids": df.loc[test_mask_random, "mp_id"].tolist(),
        "train_original_indices": df.loc[train_mask_random, "original_index"].to_numpy(),
        "test_original_indices": df.loc[test_mask_random, "original_index"].to_numpy(),
        "y_train_log_klat": df.loc[train_mask_random, "log_klat"].to_numpy(),
        "y_test_log_klat": df.loc[test_mask_random, "log_klat"].to_numpy(),
    }
    dump_split_payload(random_split, Path("processed_splits/random_split.pkl"))

    # Space group disjoint split
    train_mask_sg, test_mask_sg, train_groups, test_groups = split_by_space_group(df)
    space_group_split = {
        "split_type": "space_group_disjoint",
        "train_mp_ids": df.loc[train_mask_sg, "mp_id"].tolist(),
        "test_mp_ids": df.loc[test_mask_sg, "mp_id"].tolist(),
        "train_original_indices": df.loc[train_mask_sg, "original_index"].to_numpy(),
        "test_original_indices": df.loc[test_mask_sg, "original_index"].to_numpy(),
        "y_train_log_klat": df.loc[train_mask_sg, "log_klat"].to_numpy(),
        "y_test_log_klat": df.loc[test_mask_sg, "log_klat"].to_numpy(),
        "train_space_groups": sorted(train_groups),
        "test_space_groups": sorted(test_groups),
    }
    dump_split_payload(space_group_split, Path("processed_splits/space_group_split.pkl"))

    plot_space_group_highlight(
        df=df,
        train_groups=train_groups,
        test_groups=test_groups,
        output_path=Path("figures/space_group_split_highlight.png"),
    )

    train_mask_ood, test_mask_ood, train_threshold, test_threshold = build_ood_split(df)
    ood_split = {
        "split_type": "ood_low_klat",
        "train_mp_ids": df.loc[train_mask_ood, "mp_id"].tolist(),
        "test_mp_ids": df.loc[test_mask_ood, "mp_id"].tolist(),
        "train_original_indices": df.loc[train_mask_ood, "original_index"].to_numpy(),
        "test_original_indices": df.loc[test_mask_ood, "original_index"].to_numpy(),
        "y_train_log_klat": df.loc[train_mask_ood, "log_klat"].to_numpy(),
        "y_test_log_klat": df.loc[test_mask_ood, "log_klat"].to_numpy(),
        "train_threshold": train_threshold,
        "test_threshold": test_threshold,
    }
    dump_split_payload(ood_split, Path("processed_splits/ood_split.pkl"))

    plot_klat_histogram(
        train_values=df.loc[train_mask_ood, "klat"].to_numpy(),
        test_values=df.loc[test_mask_ood, "klat"].to_numpy(),
        output_path=Path("figures/ood_klat_histogram.png"),
    )

    print("Random 80/20 split:")
    print(f"  Train samples: {train_mask_random.sum()} | Test samples: {test_mask_random.sum()}")
    print(f"  Sample overlap: {len(set(df.loc[train_mask_random, 'mp_id']) & set(df.loc[test_mask_random, 'mp_id']))}")

    print("Space-group split:")
    print(f"  Train samples: {train_mask_sg.sum()} | Test samples: {test_mask_sg.sum()}")
    print(f"  Unique train space groups: {len(train_groups)} | Unique test space groups: {len(test_groups)}")

    print("OOD split:")
    print(f"  Train samples: {train_mask_ood.sum()} | Test samples: {test_mask_ood.sum()}")
    print(f"  Excluded samples in gap [{train_threshold:.2f}, {test_threshold:.2f}]: {len(df) - train_mask_ood.sum() - test_mask_ood.sum()}")
    print(
        f"  Thresholds -> Test: klat < {test_threshold:.2f}, Train: klat > {train_threshold:.2f}"
    )


if __name__ == "__main__":
    main()
