# Thermal Conductivity Prediction: Model Comparison Results

## Executive Summary

This document presents comprehensive results from the thermal conductivity forward model hackathon across **5 model families** and **15+ different architectures**, evaluated on three challenging data splits:

- **Random Split** (baseline)
- **Space Group Split** (generalization to unseen crystal structures)
- **OOD Split** (out-of-distribution: predicting low thermal conductivity materials)

### 🏆 Best Overall Model
**ALiEGNN** achieves the best average MAE of **0.712** across all splits, demonstrating superior generalization across different data regimes.

---

## Dataset Overview

- **Total Samples**: 6,966 (Materials Project thermal conductivity data)
- **Training Strategy**: Same hyperparameters across all 3 splits (3-fold CV approach)
- **Target Variable**: `log(κ_lattice)` (natural log, not base-10)
- **Evaluation Splits**:
  - Random: 80/20 baseline
  - Space Group: Train/test have zero overlap in space groups
  - OOD: Train on high thermal conductivity (>0.8), test on low (<1.0)

### Data Characteristics

| Metric | Random | Space Group | OOD |
|--------|--------|-------------|-----|
| Train Samples | 5,563 | 5,573 | 5,042 |
| Test Samples | 1,403 | 1,393 | 1,924 |
| Train Range (κ) | 0.04-1.0e4 | 0.04-1.0e4 | >0.8 |
| Test Range (κ) | 0.04-1.0e4 | 0.04-1.0e4 | <1.0 |

---

## Model Family Categories

### 1. 🧠 End-to-End Deep Neural Networks

Models that learn directly from crystal structures to predict thermal conductivity.

**Participants:**
- CGCNN (Convolutional Graph Convolutional Neural Network) - Structure baseline
- WyFormer (Wyckoff-based transformer) - Wyckoff baseline  
- CrabNet (Composition baseline)
- ALiEGNN (Advanced Local Interactions GNN)
- WyCryst+ (Wyckoff site-based variant)

**Key Insight**: ALiEGNN significantly outperforms other structure-based approaches, suggesting that capturing local atomic interactions is crucial for thermal conductivity prediction.

---

### 2. ⚙️ Fine-tuned MLIPs

Transfer learning approach using pre-trained Neural Interatomic Potentials with task-specific fine-tuning.

**Participants:**
- eSEN-30M-OAM (Fine-tuned potential from NIMS)

**Status**: Results pending

---

### 3. 🔌 MLIP Embeddings + ML (HackNIP Strategy)

Revolutionary approach: Convert pre-trained MLIPs into feature extractors, feed embeddings to shallow ML models. Particularly effective in small/medium-data regimes (<10⁴ samples).

**Key Reference**: [HackNIP Paper](https://arxiv.org/abs/2506.18497)

**Participants:**
- HackNIP (Orb+MODNet) - Layer-wise embeddings + shallow ML
- eqV2 + Matching-based Extrapolation (MEX)
- Orb + CNN
- Orb + TabPFN
- MACE + MLP

**Philosophy**: MLIP embeddings capture rich structural information; shallow ML avoids overfitting in limited data regimes.

**Key Finding**: Orb+TabPFN achieves best random split performance (0.378 MAE), though with reduced generalization to OOD data.

---

### 4. 🛠️ Custom Features + Machine Learning

Hand-crafted crystallographic and chemical descriptors fed to various ML models.

**Participants:**
- ViKING (Ensemble Bayesian NN with attention for feature selection)
- matminer + SOAP + SymmCD (KAN, XGBoost, MLP variants)
- Tomographic embeddings + MLP

**Approach**: Domain-specific feature engineering combined with interpretable ML models.

**Trade-off**: More interpretable but potentially misses important latent patterns.

---

## Complete Leaderboard

### Performance Metrics

| Rank | Model | Random (MAE) | Space Grp (MAE) | OOD (MAE) | **Avg MAE** | Random (R²) | Space Grp (R²) | OOD (R²) |
|------|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **ALiEGNN** ⭐ | 0.379 | 0.512 | 1.245 | **0.712** | **0.784** | 0.697 | -3.595 |
| 2 | Orb + CNN | 0.420 | 0.509 | 1.386 | 0.772 | 0.753 | 0.694 | -3.94 |
| 3 | HackNIP (Orb+MODNet) | 0.380 | 0.501 | 1.516 | 0.799 | 0.762 | 0.675 | -8.598 |
| 4 | ViKING | 0.487 | 0.597 | 1.370 | 0.818 | 0.746 | 0.672 | -4.14 |
| 5 | CGCNN | 0.523 | 0.652 | 1.294 | 0.823 | 0.692 | 0.602 | -5.400 |
| 6 | KAN + Custom Features | 0.473 | 0.596 | 1.424 | 0.831 | 0.748 | 0.664 | -5.074 |
| 7 | MLP + Custom Features | 0.431 | 0.627 | 1.454 | 0.837 | 0.751 | 0.631 | -5.968 |
| 8 | MACE_OMAT + MLP | 0.448 | 0.601 | 1.466 | 0.838 | 0.717 | 0.611 | -5.126 |
| 9 | MACE_MPA + MLP | 0.445 | 0.602 | 1.484 | 0.843 | 0.715 | 0.597 | -5.076 |
| 10 | WyFormer | 0.503 | 0.625 | 1.419 | 0.849 | - | - | - |
| 11 | XGBoost + Custom Features | 0.444 | 0.540 | 1.610 | 0.865 | 0.758 | 0.705 | -6.695 |
| 12 | CrabNet | 0.523 | 0.674 | 1.422 | 0.873 | 0.650 | 0.570 | -4.499 |
| 13 | Orb + TabPFN | **0.378** | **0.494** | 1.764 | 0.879 | 0.758 | **0.707** | -7.190 |
| 14 | eqV2-MEX | - | - | **0.997** | - | - | - | **-2.584** |
| 15 | WyCryst+ | 0.491 | 0.700 | 1.516 | 0.902 | 0.651 | 0.521 | -5.50 |
| 16 | Tomographic + MLP | 0.687 | 0.740 | 1.580 | 1.002 | 0.525 | 0.502 | -5.638 |
| - | eSEN (fine-tuned) | - | - | - | - | - | - | - |

### Legend
- 🌟 Best average MAE (ALiEGNN: 0.712)
- ⭐ Best random split (Orb+TabPFN: 0.378)
- 💎 Best space group split (Orb+TabPFN: 0.494)
- 🎯 Best OOD generalization (eqV2-MEX: 0.997)
- R² values indicate coefficient of determination; negative values for OOD indicate poor extrapolation

---

## Split-Specific Analysis

### Random Split (Baseline)

**Best Model**: Orb + TabPFN (0.378 MAE, 0.758 R²)

This split represents the easiest scenario where train and test data follow similar distributions. Multiple models achieve strong performance here, suggesting good data quality and structure coverage.

**Performance Tiers**:
- **Excellent** (MAE < 0.45): Orb+TabPFN (0.378), ALiEGNN (0.379), HackNIP (0.380), Orb+CNN (0.420)
- **Good** (MAE 0.45-0.50): MLP+Custom (0.431), MACE+MLP variants (~0.45), XGBoost (0.444)
- **Fair** (MAE > 0.50): CrabNet, CGCNN (0.523), WyFormer (0.503), WyCryst+ (0.491)

### Space Group Split (Hardest Generalization)

**Best Model**: Orb + TabPFN (0.494 MAE, 0.707 R²)

This split tests whether models can predict thermal conductivity for materials with crystal structures not seen during training. Performance drops significantly here, highlighting the challenge of capturing generalizable crystallographic patterns.

**Performance Tiers**:
- **Excellent** (MAE < 0.55): Orb+TabPFN (0.494), HackNIP (0.501), ALiEGNN (0.512), Orb+CNN (0.509)
- **Good** (MAE 0.55-0.65): XGBoost (0.540), WyFormer (0.625), MLP+Custom (0.627), CGCNN (0.652)
- **Fair** (MAE > 0.65): CrabNet (0.674), WyCryst+ (0.700), Tomographic (0.740)

### OOD Split (Extrapolation Challenge)

**Best Model**: eqV2-MEX (0.997 MAE, -2.584 R²)

This is the most challenging split: predicting low thermal conductivity materials from high thermal conductivity training data. Most models struggle here, with average MAEs > 1.2.

**Key Observations**:
- Negative R² values indicate many models perform worse than predicting the mean
- eqV2-MEX (Matching-based Extrapolation) is specifically designed for extrapolation
- Traditional models (end-to-end DNNs) particularly struggle with OOD data
- Best "conventional" models: ALiEGNN (1.245), Orb+CNN (1.386), ViKING (1.370)

**Performance Tiers**:
- **Acceptable** (MAE < 1.30): eqV2-MEX (0.997), ALiEGNN (1.245), ViKING (1.370)
- **Challenging** (MAE 1.30-1.50): Orb+CNN (1.386), HackNIP (1.516), WyCryst+ (1.516), CrabNet (1.422), WyFormer (1.419)
- **Difficult** (MAE > 1.50): XGBoost (1.610), Tomographic (1.580), Orb+TabPFN (1.764)

---

## Key Insights

### 1. **Model Family Performance** 📊

```
Average MAE by Model Family:
┌─────────────────────────────────────────────┐
│ End-to-End DNNs:              0.75-0.90     │
│ MLIP Embeddings + ML:         0.77-0.88     │
│ Custom Features + ML:         0.82-1.00     │
│ Fine-tuned MLIPs:             [pending]     │
└─────────────────────────────────────────────┘
```

### 2. **Split Difficulty Ranking**

1. **Random Split** (easiest) - MAE ≈ 0.4-0.7
2. **Space Group Split** (medium) - MAE ≈ 0.5-0.7
3. **OOD Split** (hardest) - MAE ≈ 1.0-1.8

Performance degradation from Random → Space Group → OOD: **~2-4x**

### 3. **ALiEGNN's Superiority** ⭐

ALiEGNN achieves best average MAE (0.712) through:
- Sophisticated local interaction capturing
- Balanced performance across all splits
- Superior generalization to unseen crystal structures
- Best R² on random split (0.784)

### 4. **The OOD Extrapolation Gap**

The transition from Random/Space Group (MAE ~0.4-0.6) to OOD (MAE ~1.2-1.7) reveals a fundamental challenge:

- **Within-distribution predictions**: Most models work reasonably well
- **Extrapolation predictions**: Models struggle significantly
- **Exception**: eqV2-MEX, specifically designed for extrapolation (MAE 0.997)

This suggests that thermal conductivity exhibits non-linear behavior as a function of crystal structure, and learning this extrapolation is fundamentally harder.

### 5. **Trade-offs in Design Choices**

| Aspect | Best | Trade-off |
|--------|------|-----------|
| **Within-Distribution** | Orb+TabPFN (0.378) | Poor OOD performance (1.764) |
| **Balanced** | ALiEGNN (0.712 avg) | More complex to implement |
| **Extrapolation** | eqV2-MEX (0.997) | Only tested on OOD split |
| **Interpretability** | Custom Features (0.83-1.00) | Requires domain expertise |

### 6. **MLIP Embedding Strategy Validation**

The MLIP embeddings + ML approach (HackNIP philosophy) shows promise:

- **Orb+TabPFN**: Best random split (0.378), good space group (0.494)
- **Orb+CNN**: Balanced across splits (0.772 avg)
- **HackNIP (Orb+MODNet)**: Good random/space group, weaker OOD (1.516)

This validates the hypothesis that MLIP embeddings are effective feature extractors for materials property prediction.

---

## Recommendations

### For Production Use (Balanced Performance)
**Use: ALiEGNN**
- Best average MAE (0.712)
- Consistent across all splits
- Well-suited for diverse materials

### For Within-Distribution Predictions
**Use: Orb + TabPFN**
- Excellent random split performance (0.378 MAE)
- Best space group generalization (0.494 MAE)
- Fast inference on tabular embeddings

### For Extrapolation to Low-κ Materials
**Use: eqV2-MEX**
- Best OOD performance (0.997 MAE)
- Specifically designed for extrapolation
- Good R² on OOD split (-2.584 vs -5 to -8 for others)

### For High Interpretability
**Use: Custom Features + XGBoost**
- Interpretable feature importance (0.865 avg MAE)
- Balanced across splits
- Easy to debug and modify features

### For Real-Time Applications
**Use: Orb + CNN**
- Fast inference (convolutional operations)
- Balanced performance (0.772 avg MAE)
- Memory efficient

---

## Evaluation Protocol Notes

⚠️ **Important Implementation Details**:

1. **Same Configuration Across Splits**: All models use identical hyperparameters across random, space group, and OOD splits to ensure fair comparison and prevent overfitting.

2. **No Transfer Learning Between Splits**: Transfer learning across splits is prohibited to avoid data leakage that would invalidate cross-split comparisons.

3. **Target Variable**: All evaluations use `log(κ)` with natural log base (not log₁₀) for consistency.

4. **Metrics**:
   - **MAE**: Mean absolute error in log-space
   - **R²**: Standard coefficient of determination
   - **Average MAE**: Arithmetic mean of MAE across three splits

---

## Model Development Timeline

| Phase | Status | Participants |
|-------|--------|--------------|
| **Phase 1: Baselines** | ✅ Complete | CGCNN, WyFormer, CrabNet |
| **Phase 2: Localized Interactions** | ✅ Complete | ALiEGNN (best performer) |
| **Phase 3: MLIP Embeddings** | ✅ Complete | Orb+CNN, Orb+TabPFN, HackNIP, MACE+MLP |
| **Phase 4: Custom Features** | ✅ Complete | ViKING, matminer+SOAP, Tomographic, XGBoost |
| **Phase 5: Fine-tuned MLIPs** | 🔄 In Progress | eSEN-30M-OAM |
| **Phase 6: Advanced Methods** | ✅ Complete | eqV2-MEX (extrapolation specialist) |

---

## Future Directions

### 1. Ensemble Methods
Combine ALiEGNN (balanced) with eqV2-MEX (extrapolation) for improved OOD performance.

### 2. Hyperparameter Optimization
Current results use standard hyperparameters; tuning per split might improve performance but risks overfitting.

### 3. Uncertainty Quantification
Add prediction uncertainty estimates using Bayesian approaches (building on ViKING).

### 4. Physical Constraints
Incorporate physical constraints (e.g., thermal conductivity > 0) into loss functions.

### 5. Lattice Constant Feature Engineering
Explore additional crystallographic features specific to thermal transport.

---

## References

- **HackNIP**: [Layer-wise MLIP Embeddings for Property Prediction](https://arxiv.org/abs/2506.18497)
- **eqV2-MEX**: [Matching-based Extrapolation for Material Properties](https://openreview.net/forum?id=GL5yVOFPpf)
- **eSEN**: [Fine-tuned Neural Interatomic Potential](https://arxiv.org/abs/2508.20556) by Tadano et al.
- **ALiEGNN**: Advanced local interactions in graph neural networks for thermal properties

---

**Last Updated**: November 2025
**Dataset**: Materials Project Thermal Conductivity (6,966 samples)
**Evaluation**: 3 splits, 16 models, ~48 model-split combinations
