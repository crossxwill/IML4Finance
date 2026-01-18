---
name: autogluon-tabularpredictor-set-decision-threshold
description: Set binary decision thresholds with TabularPredictor.set_decision_threshold and understand its single argument; depends on autogluon-tabularpredictor-calibrate-decision-threshold or autogluon-tabularpredictor-predict-proba for threshold selection.
---

# set_decision_threshold

## Purpose
Explain how to update the decision threshold used to convert probabilities into class predictions for binary classification.

## Usage
- “set decision threshold AutoGluon”
- “TabularPredictor.set_decision_threshold arg”

## Instructions
1. Verify `problem_type` is `binary`.
2. Choose a threshold (often from `calibrate_decision_threshold`).
3. Set the predictor threshold for future `predict()` calls.

## Function Arguments
- `decision_threshold` (float): Value in [0, 1] used to classify positive vs negative.

## Notes
- Affects only binary classification predictions; does not change probabilities.
