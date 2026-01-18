---
name: autogluon-tabular-indepth-tutorial
description: Consolidated AutoGluon Tabular tutorial summary (essentials + how-it-works + in-depth), explaining the workflow and every highlighted argument; depends on autogluon-tabularpredictor-class and autogluon-tabularpredictor-fit, plus autogluon-tabularpredictor-calibrate-decision-threshold/set-decision-threshold for threshold tuning.
---

# Tabular In-Depth Tutorial

## Purpose
Summarize advanced training, calibration, inference, and deployment options and explain the specific arguments discussed in the in-depth tutorial.

## Usage
- “AutoGluon tabular in-depth”
- “stacking bagging decision threshold calibration”
- “AutoGluon tabular essentials”
- “how AutoGluon ensembles models”

## Instructions
1. Start with defaults, then adjust `presets` and `eval_metric`.
2. Enable bagging/stacking or `auto_stack` for performance.
3. Calibrate decision thresholds for binary metrics if needed.
4. Optimize inference with `refit_full`, `persist`, and `infer_limit`.

## Essentials Workflow Summary
1. Load data with `TabularDataset`.
2. Train with `TabularPredictor(...).fit(...)`.
3. Predict, evaluate, and inspect `leaderboard()`.
4. Reload with `TabularPredictor.load(...)` when needed.

## How-It-Works Concepts (Bagging/Stacking)
- Bagging uses K-fold out-of-fold predictions to reduce variance.
- Stacking adds layers of models using OOF predictions as features, capped by a weighted ensemble.

## Function Arguments (from the tutorial)
### TabularPredictor(...)
- `label` (str): Target column.
- `eval_metric` (str | Scorer | None): Metric to optimize.
- `path` (str | Path | None): Save directory for models.
- `positive_class` (str | int | None): Positive class label for binary tasks.

### fit(...)
- `train_data` (DataFrame | str): Training data.
- `time_limit` (float | None): Training time budget.
- `presets` (str | list | dict | None): Quality preset (e.g., `best`).
- `auto_stack` (bool): Auto-enable bagging/stacking.
- `num_bag_folds` (int | None): K-fold bagging folds.
- `num_bag_sets` (int | None): Repeated bagging rounds.
- `num_stack_levels` (int | None): Number of stack layers.
- `calibrate_decision_threshold` (bool | str): Auto-calibrate threshold post-fit for binary tasks.
- `infer_limit` (float | None): Per-row inference time constraint.
- `infer_limit_batch_size` (int | None): Batch size for inference timing.
- `excluded_model_types` (list | None): Skip specific model families.
- `feature_generator` (AbstractFeatureGenerator | None): Custom preprocessing pipeline.

### calibrate_decision_threshold(...)
- `metric` (str | Scorer | None): Metric to optimize.
- `data` (DataFrame | str | None): Calibration data (optional).

### set_decision_threshold(...)
- `decision_threshold` (float): Value in [0,1] for positive class cutoff.

### predict(...)
- `data` (DataFrame | str): Data to predict.
- `model` (str | None): Model to use instead of default.
- `decision_threshold` (float | None): Override threshold for this call only.

### predict_proba(...)
- `data` (DataFrame | str): Probability prediction input.
- `as_multiclass` (bool): Binary output layout.

### leaderboard(...)
- `data` (DataFrame | str | None): Evaluation data.
- `only_pareto_frontier` (bool): Filter to speed/accuracy frontier (used with extra ensembles).
- `extra_metrics` (list | None): Additional metrics to report.

### feature_importance(...)
- `data` (DataFrame | str): Data for permutation importance.

### persist(...)
- `models` (list | str | None): Which models to keep in memory (`all`, `best`, or list).

### unpersist()
- No args; releases persisted models from memory.

### refit_full(...)
- `model` (str | None): Specific model to refit; `None` refits all eligible models.

### fit_weighted_ensemble(...)
- `expand_pareto_frontier` (bool): Create alternative ensembles for speed/accuracy tradeoffs.

### delete_models(...)
- `models_to_delete` (list): Remove unwanted models.

### save_space()
- No args; removes auxiliary artifacts to reduce disk usage.
