---
name: monotonic-constraints-using-dictionaries
description: Define and apply monotonic constraints in AutoGluon using a constraints dictionary and a feature-ordered list for boosting models; depends on autogluon-tabularpredictor-fit for passing hyperparameters.
---

# Monotonic Constraints Using Dictionaries

## Purpose
Apply monotonic constraints that align model behavior with business logic by mapping feature names to constraint directions and propagating them to AutoGluon hyperparameters.

## Usage
- "apply monotonic constraints in AutoGluon"
- "build LightGBM monotone_constraints list"
- "use dict constraints for CatBoost"

## Instructions
1. Define a constraints dictionary with values `1` for non-decreasing and `-1` for non-increasing.
2. Derive the full feature list from the trained predictor or the training DataFrame.
3. Build a feature-ordered list for GBM style models using `./scripts/build_monotone_constraints.py`.
4. Update AutoGluon hyperparameters:
   - Use the ordered list for `GBM`.
   - Use the dictionary for `CAT`.
   - Use the dictionary for `XGB`
5. Retrain the model with the updated hyperparameters.
6. Summarize the final constraints using `./templates/constraints_summary.md`.
