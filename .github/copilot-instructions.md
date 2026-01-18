# Copilot Instructions for IML4Finance

## Dependencies

The project uses conda-forge (conda) to manage `uv` and `nodejs` dependencies. If you want to run `conda`, `npm`, or `uv` commands, you must first activate the correct conda environment by running:

```bash
conda activate env_uv_202601
```

If the environment has not yet been created, you can create it by running:

```bash
conda env create -f environment_uv_202601.yml
```

If a user is unable to execute any `conda` commands, direct the user Miniforge URL: [https://github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge).

Python dependencies are managed using `uv` (installed inside a conda environment). To install Python dependencies, run:

```bash
uv sync
```

If you want to add new Python dependencies, you can do so by running:

```bash
uv add <package-name>
```

Avoid `conda install` or `pip install` for managing Python dependencies, as this may lead to inconsistencies in the `uv` environment.

## Project Purpose

The repo is for a graduate-level course called "Interpretable Machine Learning for Finance" (IML4Finance). The course focuses on teaching students how to apply interpretable machine learning techniques to financial data and problems. The course covers various topics, including feature engineering, model selection, evaluation metrics, and interpretability methods specific to finance.

## Project Frameworks and Libraries

The project primarily uses:

* AutoGluon (tabular models and AutoML)
* DuckDB (Parquet querying)
* ydata-profiling (EDA reports)
* SHAP (model explainability)
* dice-ml (counterfactuals)
* feature-engine (feature engineering)
* imbalanced-learn (sampling and imbalance handling for fraud detection)
* pandas / polars (data wrangling)
* plotnine / yellowbrick (visualization)
* Quarto (lecture/lab rendering)

## Repository Structure Notes

* Data files and loaders live in `Data/` (organized by dataset).
* Parquet and data files are not part of GitHub or Git; students should download data from the course management system (Canvas).
* Lecture and lab sources are in `Lectures/` as `.qmd` files; rendered `.html` files are generated artifacts.

## Editing Guidelines

* Prefer editing `.qmd` or `.py` sources instead of generated `.html` files.
* Keep outputs and model artifact folders unchanged unless explicitly requested.
* Avoid Jupyter notebooks (`.ipynb` files); use `.py` or `.qmd` files for code instead.