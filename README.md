# Forward Models for Thermal Conductivity Prediction

@[Jianghai](https://github.com/Ocean-JH)

This branch contains implementations of various forward models used to predict thermal conductivity in materials science.

## Table of Contents


### Features
All models share the same feature set for consistency in comparison. The features include:
- matminer composition features
- SOAP structure features
- Space group features (SymmCD representation)

The feature extraction is done by the script: `feature/featurization.py`.