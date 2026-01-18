---
name: autogluon-tabularpredictor-save
description: Save TabularPredictor artifacts with TabularPredictor.save, detailing all arguments and persistence behavior; depends on autogluon-tabularpredictor-fit and pairs with autogluon-tabularpredictor-load.
---

# save

## Purpose
Describe how to persist a fitted predictor to disk and what the save operation does.

## Usage
- “TabularPredictor.save arguments”
- “save AutoGluon predictor artifacts”

## Instructions
1. Ensure `predictor.path` points to the desired storage location.
2. Call `save()` to persist artifacts; `fit()` already auto-saves.

## Function Arguments
- `silent` (bool): If `True`, suppresses save logging.

## Notes
- `fit()` automatically saves the predictor to `path`.
