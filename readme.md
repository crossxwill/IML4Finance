# Interpretable Machine Learning for Finance

Northwestern University

[MLDS 490-0-1 Special Topics](https://www.mccormick.northwestern.edu/machine-learning-data-science/curriculum/descriptions/mlds-490-finance.html)

# Syllabus and Data

Syllabus and data sets will be available on the course Canvas.

# Textbooks

Students are highly encouraged to purchase the following two books for the course:

-   Molnar, Christoph. [*Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*](https://leanpub.com/interpretable-machine-learning/). 3rd ed. https://christophm.github.io/interpretable-ml-book/. Accessed on March 28, 2025.

-   Molnar, Christoph. [*Interpreting Machine Learning Models With SHAP: A Guide With Python Examples and Theory on Shapley Values*](https://christophmolnar.com/books/shap/). Independent publication, 2023.

# Key Repo Files

-   `conda_env_iml4finance_2026.yml` : cross-platform conda environment for the course

-   `IML4Finance.code-workspace`: workspace file for the local repository

-	`.vscode/Interpretable ML.code-profile`: VS code profile of extensions and keyboard shortcuts

# System Prerequisites

-   [Miniforge3](https://github.com/conda-forge/miniforge)

-   [VS Code IDE](https://code.visualstudio.com/)

-   [Git](https://git-scm.com/)

-   [GitHub Desktop](https://github.com/apps/desktop)

# Clone the Repo

1.  Open GitHub Desktop (and link your GitHub account).
2.  Click on `File` \> `Clone Repository...`.
3.  Select the `URL` tab.
4.  Enter the URL of the repository
5.  Click on `Choose...` and select the directory where you want to save the repository.
6.  Click on `Clone`.

# Conda Environment

1.  Open a terminal.

	On Windows, use `Miniforge Prompt` or a shell where `conda` is initialized.

	On macOS or Linux, use your regular terminal.

2.  Change the directory to the local repository (see Step 5 in the previous section).

3.  Create the conda environment from `conda_env_iml4finance_2026.yml`:

``` bash
conda env create -f conda_env_iml4finance_2026.yml
```

`conda_env_iml4finance_2026.yml` is the only supported environment file for this repo.

4.  Activate the conda environment:

``` bash
conda activate env_iml4finance_2026
```

5.  Launch VS Code:

``` bash
code
```

Lecture, lab, and rendered output filenames use underscore-based names such as `Lecture_02.qmd`, `Lab_01.qmd`, and `Lecture_02.html` to avoid shell and Quarto issues caused by spaces in file paths.

# Repository Structure

-   `Lectures/`: lecture source files (`.qmd`), rendered lecture outputs (`.html`, `_files/`), lecture PDFs, lecture model folders, `Images/`, and `references.bib`

-   `Labs/`: lab source files (`Lab_01.qmd` through `Lab_04.qmd`), rendered lab outputs (`.html`, `_files/`), EDA reports, and lab model folders

-   `Quizzes/`: quiz scripts and quiz-specific data files

-   `Examples/`: standalone example spreadsheets used in the course

-   `course_utils/`: shared Python utilities for labs and other course materials

-   `Data/`: dataset-specific loaders, documentation, and local data assets

Labs now import shared utilities from `course_utils.helpers`, while lectures continue to use local lecture assets from `Lectures/`.

Rendered Quarto outputs now live beside their source `.qmd` files in `Labs/` and `Lectures/`. Model folders created by the course materials also live beside the `.qmd` files that generate or consume them.

# VS Code Profile

1.  Open VS Code
2.  Type `Ctrl + Shift + P` on Windows and `Cmd + Shift + P` on Mac
3.  Type `>Profiles: New Profile`
4.  Delete the profile that is called "Untitled"
5.  Click on the drop-down arrow next to the "New Profile" button
6.  Click `Import Profile`
7.  Click `Select File`
8.  Choose `Interpretable ML.code-profile`

# Hidden folders on a Mac

To see hidden folders (that start with a dot) on a Mac, toggle on/off with `Cmd + Shift + .`.