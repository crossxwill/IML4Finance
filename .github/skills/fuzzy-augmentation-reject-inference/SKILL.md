---
name: fuzzy-augmentation-reject-inference
description: Build a Through-the-Door training set with reject inference using fuzzy augmentation, including PD-based sample weights; pairs with autogluon-tabularpredictor-fit for modeling the augmented data.
---

# Fuzzy Augmentation Reject Inference

## Purpose
Create a weighted Through-the-Door dataset by scoring rejected applicants with a KGB model, duplicating them as good and bad outcomes, and assigning `sample_weight` by predicted PD.

## Usage
- "apply fuzzy augmentation"
- "build TTD dataset with reject inference"
- "weight rejected applicants by PD"

## Instructions
1. Train a logistic regression model on accepted data to estimate PD.
2. Score rejected applicants to get PD values.
3. Create two copies of rejected rows:
   - Copy A: `default_flag = 1`, `sample_weight = PD`
   - Copy B: `default_flag = 0`, `sample_weight = 1 - PD`
4. Combine accepted data with both copies of rejected data and add a `source` column.
5. Use `./scripts/create_ttd_data.py` to standardize the augmentation.
6. Summarize results using `./templates/ttd_summary.md`.
