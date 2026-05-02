---
name: ydata-eda-profiling
description: Generate and compare ydata-profiling EDA reports with sampling, consistent random seeds, and HTML outputs; often follows duckdb-parquet-lab-workflow when data is queried from Parquet.
---

# YData Profiling EDA

## Purpose
Create consistent EDA reports for train and test or accepted and rejected datasets using `ProfileReport`, including comparisons and saved HTML outputs.

## Usage
- "generate ydata-profiling report"
- "compare train and test EDA"
- "create HTML EDA report"

## Instructions
1. Set a sampling fraction for large datasets and a fixed random seed.
2. Create `ProfileReport` objects with `progress_bar=False`, and disable duplicates and interactions when speed matters.
3. Compare reports using `.compare()` and save to HTML with `.to_file()`.
4. Use `./scripts/generate_eda_report.py` for consistent report creation.
5. Use `./templates/eda_compare_block.md` to document inputs, sample fractions, and output paths.
