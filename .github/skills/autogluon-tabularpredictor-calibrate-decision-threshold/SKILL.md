---
name: autogluon-tabularpredictor-calibrate-decision-threshold
description: Calibrate binary classification decision thresholds with TabularPredictor.calibrate_decision_threshold, detailing every argument and trade-off; depends on autogluon-tabularpredictor-fit and pairs with autogluon-tabularpredictor-set-decision-threshold and autogluon-tabularpredictor-predict-proba.
---

# calibrate_decision_threshold

## Purpose
Explain how to calibrate decision thresholds for binary classification and how each argument controls the search, data usage, and output.

## Usage
- “calibrate decision threshold AutoGluon”
- “TabularPredictor.calibrate_decision_threshold args”

## Instructions
1. Verify `problem_type` is `binary`.
2. Choose metric and model for calibration.
3. Decide search granularity and optional secondary refinement.
4. Apply the returned threshold via `set_decision_threshold`.

## Function Arguments
- `data` (DataFrame | str | None): Dataset for calibration. Must include the label column. If `None`, uses internal validation or OOF data.
- `metric` (str | Scorer | None): Metric to optimize. If `None`, uses `predictor.eval_metric`.
- `model` (str): Model name to use for probability outputs. Use `best` for the current best model.
- `decision_thresholds` (int | list[float]): If int, number of thresholds on either side of 0.5 to scan; if list, explicit thresholds to evaluate.
- `secondary_decision_thresholds` (int | None): Optional fine-grained search around the best threshold from the first pass; set `None` to skip.
- `subsample_size` (int | None): Cap rows used for calibration to speed up; set `None` to disable subsampling.
- `verbose` (bool): If `True`, log calibration details.

## Output
- Returns a float between 0 and 1 that maximizes the chosen metric on the calibration data.

## Notes
- Optimizing one metric can hurt others (e.g., `balanced_accuracy` vs `accuracy`).
