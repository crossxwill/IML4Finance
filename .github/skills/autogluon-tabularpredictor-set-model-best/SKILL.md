---
name: autogluon-tabularpredictor-set-model-best
description: Set the default prediction model with TabularPredictor.set_model_best, detailing all arguments and effects; depends on autogluon-tabularpredictor-fit and is often guided by autogluon-tabularpredictor-fit-summary or leaderboard outputs.
---

# set_model_best

## Purpose
Explain how to override the default best model used by `predict()` and `predict_proba()`.

## Usage
- “set_model_best AutoGluon”
- “choose default model for predictor”

## Instructions
1. Identify candidate models from `predictor.model_names()` or `leaderboard()`.
2. Set the desired model as the default best.
3. Persist the trainer (`save_trainer=True`) if you need the change to survive reloads. Generally recommended.

## Function Arguments
- `model` (str): Model name to use as default. Must exist and be inferable; otherwise raises `AssertionError`.
- `save_trainer` (bool): If `True`, saves the trainer so the new default persists when reloading the predictor.

## Notes
- Subsequent training operations (e.g., `fit_extra`, `refit_full`, `distill`) can overwrite the best model.
