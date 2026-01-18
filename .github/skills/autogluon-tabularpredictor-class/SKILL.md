---
name: autogluon-tabularpredictor-class
description: Explain and configure AutoGluon’s TabularPredictor constructor, including all init arguments and their effects for tabular classification/regression; prerequisite for autogluon-tabularpredictor-fit, predict-proba, save/load, fit-summary, calibrate-decision-threshold, set-decision-threshold, and set-model-best.
---

# TabularPredictor Constructor

## Purpose
Provide a precise, argument-by-argument guide for initializing `TabularPredictor`, including how each parameter affects training, evaluation, and artifact storage.

## Usage
- “TabularPredictor __init__ arguments”
- “configure TabularPredictor label/eval_metric/path”

## Instructions
1. Confirm the prediction target and problem type.
2. Map all constructor arguments to concrete values.
3. Highlight any arguments that influence evaluation, saving, or class handling.

## Constructor Arguments
- `label` (str): Target column name to predict.
- `problem_type` (str | None): Prediction task. Options: `binary`, `multiclass`, `regression`, `quantile`. If `None`, inferred from label values.
- `eval_metric` (str | Scorer | None): Metric optimized during training/ensembling. If `None`, AutoGluon chooses based on `problem_type`.
- `path` (str | Path | None): Folder for models and logs. If `None`, AutoGluon uses a timestamped folder under `AutogluonModels`.
- `verbosity` (int): Logging detail. `0` only errors, `1` warnings, `2` standard, `3` verbose, `4` very verbose.
- `log_to_file` (bool): If `True`, save logs to a file.
- `log_file_path` (str): Log file path. `auto` uses `<predictor.path>/logs/predictor_log.txt`.
- `sample_weight` (str | None): Column name for sample weights, or special values `auto_weight` / `balance_weight`.
- `weight_evaluation` (bool): If `True`, apply sample weights when evaluating metrics (requires weights in evaluation data).
- `groups` (str | None): Column name used for group-based splitting in bagging (experimental).
- `positive_class` (str | int | None): Positive class label for binary metrics like `f1`.
- `**kwargs` (advanced config):
  - `learner_type` (AbstractLearner): Overrides the learner class. Keep default unless extending internals.
  - `learner_kwargs` (dict | None): Extra learner kwargs.
  - `ignored_columns` (list | None): Columns excluded from features.
  - `label_count_threshold` (int): Minimum class frequency for multiclass labels to be retained.
  - `cache_data` (bool): Save training/validation data to disk for reuse and advanced features.
  - `trainer_type` (AbstractTabularTrainer): Overrides trainer class.
  - `default_base_path` (str | Path | None): Base path for auto timestamped folders when `path=None`.
