---
name: autogluon-tabularpredictor-predict-proba
description: Use TabularPredictor.predict_proba with full argument guidance for classification probability outputs; depends on autogluon-tabularpredictor-fit and supports autogluon-tabularpredictor-calibrate-decision-threshold/set-decision-threshold.
---

# predict_proba

## Purpose
Explain how to generate class probabilities and how each argument controls output format and preprocessing.

## Usage
- “TabularPredictor.predict_proba args”
- “AutoGluon probability predictions”

## Instructions
1. Ensure the predictor is for classification (not regression/quantile).
2. Decide output type (`pandas` vs `numpy`) and binary layout.
3. Use `transform_features=False` only when you already transformed data.

## Function Arguments
- `data` (DataFrame | str): Input data or file path; may include label column (ignored for prediction).
- `model` (str | None): Model name to use; `None` uses best model.
- `as_pandas` (bool): If `True`, return DataFrame/Series; otherwise ndarray.
- `as_multiclass` (bool): For binary tasks, `True` returns two columns (neg/pos). `False` returns positive class only.
- `transform_features` (bool): If `True`, apply global preprocessing. Set `False` if data was already transformed via `transform_features()`.

## Output
- Probabilities per row; DataFrame columns align with `predictor.class_labels` when multiclass or `as_multiclass=True`.
