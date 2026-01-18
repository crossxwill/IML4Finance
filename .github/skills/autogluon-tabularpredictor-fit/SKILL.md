---
name: autogluon-tabularpredictor-fit
description: Detail every TabularPredictor.fit argument, including presets, ensembling, resources, HPO, and deployment settings; depends on autogluon-tabularpredictor-class and feeds autogluon-tabularpredictor-fit-summary, predict-proba, calibrate-decision-threshold, set-model-best, save/load.
---

# TabularPredictor.fit

## Purpose
Provide a complete, argument-by-argument guide to `fit()` for training tabular models, including ensembling, resource limits, and deployment options.

## Usage
- “TabularPredictor.fit arguments”
- “AutoGluon fit presets and stacking”

## Instructions
1. Identify training data and label column.
2. Choose presets and time budget.
3. Decide ensembling and resource constraints.
4. Configure optional advanced kwargs only when necessary.

## Function Arguments
- `train_data` (DataFrame | str): Training data table or path.
- `tuning_data` (DataFrame | str | None): Optional validation data for tuning/ensembling. Avoid when bagging unless `use_bag_holdout=True`.
- `time_limit` (float | None): Wall-clock time limit in seconds for training.
- `presets` (list[str] | str | dict | None): Quality presets (e.g., `best_quality`, `good_quality`, `optimize_for_deployment`). Later presets override earlier ones.
- `hyperparameters` (dict | str | None): Model hyperparameter config or preset (`default`, `zeroshot`, `light`, etc.).
- `feature_metadata` (FeatureMetadata | str): Feature type metadata; `infer` to auto-detect.
- `infer_limit` (float | None): Per-row inference time constraint in seconds.
- `infer_limit_batch_size` (int | None): Batch size used to estimate inference time.
- `fit_weighted_ensemble` (bool): Fit a weighted ensemble per stack level.
- `fit_full_last_level_weighted_ensemble` (bool): Last-level weighted ensemble uses all prior models.
- `full_weighted_ensemble_additionally` (bool): Fit an extra ensemble after stacking.
- `dynamic_stacking` (bool | str): Automatically decide whether to use stacking to avoid stacked overfitting.
- `calibrate_decision_threshold` (bool | str): Auto-calibrate decision threshold for binary metrics needing class predictions.
- `num_cpus` (int | str): CPU budget; `auto` lets AutoGluon decide.
- `num_gpus` (int | str): GPU budget; `auto` lets AutoGluon decide.
- `fit_strategy` ("sequential" | "parallel"): Training strategy; `parallel` uses Ray (experimental, no GPU support).
- `memory_limit` (float | str): Soft memory limit in GB; `auto` uses available memory.
- `callbacks` (list[AbstractCallback] | None): Experimental callbacks to observe or alter training.
- `**kwargs` (advanced controls):
  - `auto_stack` (bool): Enable automatic bagging/stacking.
  - `num_bag_folds` (int | None): Number of bagging folds (0 disables bagging).
  - `num_bag_sets` (int | None): Repeats of bagging to reduce variance.
  - `num_stack_levels` (int | None): Stack levels (0 disables stacking).
  - `delay_bag_sets` (bool): Delay repeated bagging until later in training.
  - `holdout_frac` (float | None): Holdout fraction when no tuning data is provided.
  - `use_bag_holdout` (bool | str): Use holdout data for ensembles during bagging.
  - `hyperparameter_tune_kwargs` (str | dict | None): HPO settings (`auto`, `random`, or explicit searcher/scheduler).
  - `feature_prune_kwargs` (dict | None): Recursive feature elimination settings.
  - `ag_args` (dict | None): Shared model meta-args (naming, priority, valid problem types, etc.).
  - `ag_args_fit` (dict | None): Shared fit constraints (time limits, memory ratios, etc.).
  - `ag_args_ensemble` (dict | None): Shared ensemble controls (stacker validity, base model caps, folds).
  - `ds_args` (dict): Dynamic stacking configuration (validation procedure, detection time fraction, etc.).
  - `included_model_types` (list | None): Restrict model types to train.
  - `excluded_model_types` (list | None): Skip model types even if in hyperparameters.
  - `refit_full` (bool | str): Refit models on full data after training (`all`, `best`, or specific model).
  - `save_bag_folds` (bool): Save bag fold models (required for bagged inference unless `refit_full`).
  - `set_best_to_refit_full` (bool): Use refit-full model as default best.
  - `keep_only_best` (bool): Delete all but best model and its ancestors.
  - `save_space` (bool): Remove auxiliary artifacts to reduce disk usage.
  - `feature_generator` (AbstractFeatureGenerator): Custom preprocessing pipeline.
  - `unlabeled_data` (DataFrame | None): Reserved; currently unused.
  - `verbosity` (int): Override predictor verbosity.
  - `raise_on_model_failure` (bool): Raise immediately on model failure (debugging).
  - `raise_on_no_models_fitted` (bool): Raise if no models succeed.
  - `calibrate` (bool | str): Probability/quantile calibration (`auto` recommended).
  - `test_data` (DataFrame | str | None): Test data used only for logging/learning curves.
  - `learning_curves` (bool | dict | None): Enable learning curve collection; dict may include `metrics` and `use_error`.

## Output
- Returns the fitted `TabularPredictor` (self).
