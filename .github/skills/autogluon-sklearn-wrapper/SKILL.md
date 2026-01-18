---
name: autogluon-sklearn-wrapper
description: Build a scikit-learn compatible wrapper for AutoGluon TabularPredictor with feature name checks, sample_weight support, and predict/predict_proba methods; depends on custom-sklearn-estimator for sklearn API rules and autogluon-tabularpredictor-class/fit for predictor usage.
---

# AutoGluon sklearn Wrapper

## Purpose
Standardize AutoGluon models so they integrate with scikit-learn tools like `PartialDependenceDisplay()` while preserving feature names, weights, and fitted-state checks.

## Usage
- "wrap AutoGluon to work with sklearn"
- "create a BaseEstimator wrapper for TabularPredictor"
- "use AutoGluon with PDP or ICE"

## Instructions
1. Implement a wrapper class extending `BaseEstimator` and `ClassifierMixin`.
   - Track `feature_names_`, `n_features_in_`, `classes_`, and `is_fitted_`.
   - Provide `__sklearn_is_fitted__()` to support `check_is_fitted()`.
2. In `fit()`:
   - Convert `X` to a pandas DataFrame using stored feature names.
   - Add the label column and an optional sample weight column named by `predictor_args['sample_weight']` when present.
   - Convert to `TabularDataset` and call `TabularPredictor(...).fit(...)` using cleaned `fit_args`.
3. In `predict()` and `predict_proba()`:
   - Call `check_is_fitted(self)`.
   - Validate feature names and feature counts.
   - Convert inputs to `TabularDataset` before calling AutoGluon.
4. For numpy input in `fit()`, create `feat_0`, `feat_1`, ... names and persist them.
5. Use `./templates/wrapper_snippet.md` as the baseline layout.
