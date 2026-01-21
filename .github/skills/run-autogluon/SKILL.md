---
name: run-autogluon
description: Consolidated AutoGluon skill covering end-to-end TabularPredictor workflow (constructor, fit, predict_proba, fit_summary, save/load, set_model_best), binary threshold calibration and setting, sklearn wrapper integration, and monotonic constraints. Use for any AutoGluon Tabular questions, training/ensembling configuration, inference/probabilities, threshold tuning, deployment persistence, or sklearn interoperability.
---

# Run AutoGluon (Consolidated)

## Purpose
Provide a single, comprehensive reference for AutoGluon TabularPredictor usage in this repo, including model construction, training, evaluation, calibration, saving/loading, sklearn integration, and monotonic constraint workflows.

## Usage
- “AutoGluon TabularPredictor arguments or workflow”
- “fit/predict_proba/fit_summary for AutoGluon”
- “calibrate or set decision threshold”
- “save/load AutoGluon models”
- “set best model in AutoGluon”
- “AutoGluon sklearn wrapper or PDP compatibility”
- “monotonic constraints in AutoGluon”

## Instructions
1. Identify the task category:
   - Initialize predictor, fit, evaluate, infer, or persist.
   - Calibrate thresholds for binary classification.
   - Wrap predictor for sklearn usage.
   - Apply monotonic constraints.
2. Use the relevant section below to map arguments and steps.
3. Prefer minimal changes and align with course workflows (data in Data/, labs in Lectures/).

## Core Workflow (Essentials)
1. Load data with `TabularDataset`.
2. Train with `TabularPredictor(...).fit(...)`.
3. Evaluate with `leaderboard()` or `fit_summary()`.
4. Predict with `predict()` or `predict_proba()`.
5. Save/load using `save()` and `load()` as needed.

## TabularPredictor Constructor (Key Args)
- `label`: target column name.
- `problem_type`: `binary`, `multiclass`, `regression`, `quantile`, or inferred.
- `eval_metric`: metric to optimize; defaults to AutoGluon choice by task.
- `path`: model artifact directory; defaults to `AutogluonModels`.
- `verbosity`: 0–4 for logging.
- `log_to_file`, `log_file_path`.
- `sample_weight`: column name or `auto_weight`/`balance_weight`.
- `weight_evaluation`: use weights during evaluation.
- `groups`: group-based splitting (experimental).
- `positive_class`: label for positive class in binary.
- Advanced: `ignored_columns`, `cache_data`, `learner_type`, `trainer_type`, `default_base_path`.

## fit(...) (Key Args)
- `train_data`, `tuning_data`.
- `time_limit`, `presets`, `hyperparameters`.
- `feature_metadata`, `feature_generator`.
- Ensembling: `auto_stack`, `num_bag_folds`, `num_bag_sets`, `num_stack_levels`.
- `fit_weighted_ensemble`, `full_weighted_ensemble_additionally`.
- `dynamic_stacking`.
- Calibration: `calibrate_decision_threshold`, `calibrate`.
- Resources: `num_cpus`, `num_gpus`, `memory_limit`, `fit_strategy`.
- Deployment: `refit_full`, `set_best_to_refit_full`, `keep_only_best`, `save_space`.

## fit_summary(...)
- `verbosity`: 0+; `2`+ includes plots.
- `show_plot`: open plot in browser.
- Returns a summary dict for analysis/reporting.

## predict_proba(...)
- Classification only.
- `as_pandas`: DataFrame vs ndarray.
- `as_multiclass`: for binary, return two columns vs positive class only.
- `transform_features`: set `False` only if data already transformed.

## Threshold Calibration (Binary)
### calibrate_decision_threshold(...)
- `data`: labeled calibration data or `None` for OOF/validation.
- `metric`: optimize target metric.
- `decision_thresholds`: int or list.
- `secondary_decision_thresholds`: optional fine search.
- `subsample_size`, `verbose`.

### set_decision_threshold(...)
- `decision_threshold`: float in [0,1] for binary predictions.

## set_model_best(...)
- `model`: model name from `model_names()`/`leaderboard()`.
- `save_trainer`: persist default best across reloads.

## save/load
- `save(silent=True|False)`.
- `load(path, verbosity=None, require_version_match=True, require_py_version_match=True, check_packages=True)`.
- Only load trusted artifacts (pickle).

## sklearn Wrapper Integration
1. Implement `ClassifierMixin` + `BaseEstimator` (mixin should go first)
2. Track `feature_names_`, `n_features_in_`, `classes_`, `is_fitted_`.
3. In `fit()`:
   - Convert `X` to DataFrame.
   - Add label column and optional `sample_weight` column.
   - Call `TabularPredictor(...).fit(...)`.
4. In `predict()`/`predict_proba()`:
   - Validate feature names/counts.
   - Convert to `TabularDataset`.
5. Use `./templates/wrapper_snippet.md` as baseline layout.

## Monotonic Constraints (Dictionaries)
1. Define a constraints dict: `1` non-decreasing, `-1` non-increasing.
2. Build a feature-ordered list for GBM models.
3. Set hyperparameters:
   - `GBM`: list order.
   - `CAT`, `XGB`: dict form.
4. Retrain with updated hyperparameters.

## Notes
- Prefer course data folders under Data/ and lab scripts under Lectures/.
- Keep changes aligned with existing lab workflows.
