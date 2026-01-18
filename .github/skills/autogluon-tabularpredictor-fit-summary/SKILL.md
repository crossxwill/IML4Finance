---
name: autogluon-tabularpredictor-fit-summary
description: Generate and interpret training summaries with TabularPredictor.fit_summary, detailing all arguments and outputs; depends on autogluon-tabularpredictor-fit and complements autogluon-tabularpredictor-set-model-best.
---

# fit_summary

## Purpose
Explain how to obtain a training report and how each argument affects output verbosity and plots.

## Usage
- “TabularPredictor.fit_summary args”
- “training summary AutoGluon”

## Instructions
1. Call `fit_summary()` after training to review models, timings, and metadata.
2. Adjust verbosity to control print and plot generation.

## Function Arguments
- `verbosity` (int): `<=0` no output, `1` high-level summary, `2` summary + plots, `>=3` full details.
- `show_plot` (bool): If `True`, open the summary plot in a browser when verbosity > 1.

## Output
- Returns a dict of detailed training information (can be large).
