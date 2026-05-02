# Interpretable Machine Learning for Finance

Northwestern University

[MLDS 490-0-1 Special Topics](https://www.mccormick.northwestern.edu/machine-learning-data-science/curriculum/descriptions/mlds-490-finance.html)

## Table of Contents

- [Syllabus and Data](#syllabus-and-data)
- [Textbooks](#textbooks)
- [System Prerequisites](#system-prerequisites)
- [Quick Start](#quick-start)
- [Clone the Repo](#clone-the-repo)
- [Conda Environment](#conda-environment)
- [VS Code Profile](#vs-code-profile)
- [Keeping the Repo Up to Date](#keeping-the-repo-up-to-date)
- [Repository Structure](#repository-structure)
- [Key Repo Files](#key-repo-files)
- [Agent Skills (for GitHub Copilot)](#agent-skills-for-github-copilot)
- [Troubleshooting](#troubleshooting)
- [Hidden Folders on a Mac](#hidden-folders-on-a-mac)

## Syllabus and Data

Syllabus and data sets will be available on the course Canvas.

## Textbooks

Students are highly encouraged to purchase the following two books for the course:

- Molnar, Christoph. [*Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*](https://leanpub.com/interpretable-machine-learning/). 3rd ed. https://christophm.github.io/interpretable-ml-book/. Accessed on March 28, 2025.
  > Also available to read **free online** at the link above.

- Molnar, Christoph. [*Interpreting Machine Learning Models With SHAP: A Guide With Python Examples and Theory on Shapley Values*](https://christophmolnar.com/books/shap/). Independent publication, 2023.

## System Prerequisites

- [Miniforge3](https://github.com/conda-forge/miniforge)
- [VS Code IDE](https://code.visualstudio.com/)
- [Git](https://git-scm.com/)
- [GitHub Desktop](https://github.com/apps/desktop)

## Quick Start

If you already have the prerequisites installed:

1. **Clone** this repository (see [Clone the Repo](#clone-the-repo)).
2. **Create and activate** the conda environment (see [Conda Environment](#conda-environment)).
3. **Import the VS Code profile** (see [VS Code Profile](#vs-code-profile)).

Then open any `.qmd` file in `Lectures/` or `Labs/` to get started.

## Clone the Repo

1. Open GitHub Desktop (and link your GitHub account).
2. Click on `File` > `Clone Repository...`.
3. Select the `URL` tab.
4. Enter the URL of the repository.
5. Click on `Choose...` and select the directory where you want to save the repository.
6. Click on `Clone`.

## Conda Environment

The environment includes **Python 3.11** with key packages: AutoGluon, SHAP, dice-ml, scikit-learn, imbalanced-learn, ydata-profiling, yellowbrick, plotnine, DuckDB, Quarto, and more.

1. Open a terminal.

   On Windows, use `Miniforge Prompt` or a shell where `conda` is initialized.

   On macOS or Linux, use your regular terminal.

2. Change the directory to the local repository (see Step 5 in [Clone the Repo](#clone-the-repo)).

3. Create the conda environment from `conda_env_iml4finance_2026.yml`:

   ```bash
   conda env create -f conda_env_iml4finance_2026.yml
   ```

   `conda_env_iml4finance_2026.yml` is the only supported environment file for this repo.

4. Activate the conda environment:

   ```bash
   conda activate env_iml4finance_2026
   ```

5. Launch VS Code:

   ```bash
   code
   ```

> **Note:** Lecture, lab, and rendered output filenames use underscore-based names such as `Lecture_02.qmd`, `Lab_01.qmd`, and `Lecture_02.html` to avoid shell and Quarto issues caused by spaces in file paths.

## VS Code Profile

The profile pre-configures recommended extensions (Python, Quarto, Jupyter, etc.), workspace settings, and keyboard shortcuts for the course.

1. Open VS Code.
2. Type `Ctrl + Shift + P` on Windows / `Cmd + Shift + P` on Mac.
3. Type `>Profiles: New Profile`.
4. Delete the profile that is called "Untitled".
5. Click on the drop-down arrow next to the "New Profile" button.
6. Click `Import Profile`.
7. Click `Select File`.
8. Choose `Interpretable ML.code-profile` from the `.vscode/` folder.

## Keeping the Repo Up to Date

New materials and fixes may be pushed throughout the quarter. To pull the latest changes:

```bash
git pull
```

Run this from the repository directory **before** starting each lab or lecture.

## Repository Structure

| Folder / File | Description |
|---|---|
| `Lectures/` | Lecture source files (`Lecture_01.qmd`, `Lecture_02.qmd`), rendered HTML/PDF outputs, tracked model folders, `Images/`, and `references.bib` |
| `Labs/` | Lab source files (`Lab_01.qmd` through `Lab_04.qmd`), rendered HTML outputs, and EDA reports |
| `Quizzes/` | Quiz scripts (`quiz02-02.py`, `quiz02-04.py`, `quiz03-01.py`) and supporting data files |
| `Examples/` | Standalone example spreadsheets: `precision_recall_f1_example.xlsx`, `score_PDO_example.xlsx`, `shap_example.xlsx` |
| `course_utils/` | Shared Python utilities (`helpers.py`) imported by all labs — includes seed setting, AutoGluon sklearn wrapper, scoring functions, and plotting helpers |
| `Data/` | Dataset-specific loaders organized by dataset: `cibil/`, `credit-card-fraud/`, `fanniemae-SFLP/`, `lendingclub/` |
| `.github/` | Copilot instructions (`copilot-instructions.md`) and 25+ agent skills (see [Agent Skills](#agent-skills-for-github-copilot)) |
| `.vscode/` | Workspace settings (`settings.json`) and the importable VS Code profile |
| `.gitignore` | Specifies intentionally untracked files: CSVs, parquet, model folders, zip files, and other generated artifacts |
| `conda_env_iml4finance_2026.yml` | Cross-platform conda environment specification |
| `IML4Finance.code-workspace` | VS Code workspace file for the repository |

### Lab Summaries

| Lab | Topic | Accompanies |
|---|---|---|
| Lab 01 | Feature selection: filter, wrapper, and embedded methods | Lecture 01 |
| Lab 02 | Through-the-Door (TTD) training sets and reject inference | Lecture 02 |
| Lab 03 | Model explainability with SHAP | Lecture 03 |
| Lab 04 | Class imbalance handling: class-sensitive, cost-sensitive, and SMOTE approaches | Lecture 04 |

Labs import shared utilities from `course_utils.helpers`, while lectures use local assets from `Lectures/`.

Rendered Quarto outputs live beside their source `.qmd` files in `Labs/` and `Lectures/`. Model folders are tracked under `Lectures/`; running course materials may create additional model folders based on the execution working directory.

## Key Repo Files

- `conda_env_iml4finance_2026.yml` — cross-platform conda environment for the course
- `IML4Finance.code-workspace` — workspace file for the local repository
- `.vscode/Interpretable ML.code-profile` — VS Code profile of extensions and keyboard shortcuts
- `.gitignore` — defines which files are excluded from version control (data files, model artifacts, etc.)
- `.github/copilot-instructions.md` — Copilot configuration: conda environment, dependencies, editing guidelines, and rendering instructions
- `course_utils/helpers.py` — shared Python utilities used across all labs

## Agent Skills (for GitHub Copilot)

This repo includes 25+ agent skills in `.github/skills/` that extend GitHub Copilot with domain-specific knowledge. Key skills include:

| Skill | Purpose |
|---|---|
| `run-autogluon` | End-to-end AutoGluon TabularPredictor workflow (fit, predict, calibrate, save/load) |
| `render-quarto-markdown-files` | Render `.qmd` files from the command line |
| `edit-quarto-markdown-files` | Edit `.qmd` / `.md` files with proper nested code block handling |
| `duckdb-parquet-lab-workflow` | Query Parquet files with DuckDB and convert to pandas |
| `ydata-eda-profiling` | Generate and compare ydata-profiling EDA reports |
| `custom-sklearn-estimator` | Build scikit-learn compatible custom estimators |
| `fuzzy-augmentation-reject-inference` | Build TTD training sets with reject inference via fuzzy augmentation |
| `create-plan` | Create concise implementation plans for coding tasks |

These skills are automatically available when using Copilot in this workspace.

## Troubleshooting

### "conda not found" or "conda is not recognized"

Ensure Miniforge3 is installed and the `Miniforge Prompt` is being used on Windows. On macOS/Linux, run `conda init` and restart your shell.

### Environment creation fails

Make sure you are in the repository root directory and using the exact command:

```bash
conda env create -f conda_env_iml4finance_2026.yml
```

If a package conflict occurs, try updating conda first:

```bash
conda update -n base conda
```

### Quarto render errors

Ensure the conda environment is activated before rendering. If you see "quarto not found," verify the environment was created successfully with `conda env list`.

### Missing data files

Data files (`.csv`, `.parquet`, `.xlsx`) are **not** tracked in Git. Download them from the course Canvas and place them in the appropriate `Data/` subdirectory.

## Hidden Folders on a Mac

To see hidden folders (that start with a dot) on a Mac, toggle on/off with `Cmd + Shift + .`.
