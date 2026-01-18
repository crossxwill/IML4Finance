---
name: autogluon-tabularpredictor-load
description: Load saved predictors with TabularPredictor.load, detailing all arguments and security/version implications; depends on autogluon-tabularpredictor-save and assumes a prior autogluon-tabularpredictor-fit.
---

# load

## Purpose
Explain how to restore a saved predictor and how version checks and verbosity are controlled.

## Usage
- “TabularPredictor.load arguments”
- “load AutoGluon predictor safely”

## Instructions
1. Provide the exact predictor directory path.
2. Decide whether to enforce version and Python checks.
3. Load only trusted artifacts (pickle security).

## Function Arguments
- `path` (str): Directory containing the saved predictor.
- `verbosity` (int | None): Override predictor verbosity after loading (`0`–`4`).
- `require_version_match` (bool): Enforce AutoGluon version match.
- `require_py_version_match` (bool): Enforce Python version match (micro version mismatch warns).
- `check_packages` (bool): If `True`, warn on package version mismatches.

## Notes
- Loading uses pickle; never load untrusted or tampered artifacts.
