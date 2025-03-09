# Interpretable Machine Learning for Finance

Northwestern University

[MLDS 490-0-1 Special Topics](https://www.mccormick.northwestern.edu/machine-learning-data-science/curriculum/descriptions/mlds-490-1.html)

# Syllabus and Data

Syllabus and data sets will be available on the course Canvas.

# Textbooks

Students are highly encouraged to purchase the following two books for the course:

-   Molnar, Christoph. [*Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*](https://leanpub.com/interpretable-machine-learning/). 2nd ed. https://christophm.github.io/interpretable-ml-book. Accessed on October 25, 2024.

-   Molnar, Christoph. [*Interpreting Machine Learning Models With SHAP: A Guide With Python Examples and Theory on Shapley Values*](https://christophmolnar.com/books/shap/). Independent publication, 2023.

# Key Repo Files

-   `conda_env_requirements.yml` : requirements file that specifies the conda environment for the repository

-   `Interpretable ML.code-profile`: configuration file for VS code profile

-   `IML4Finance.code-workspace`: workspace file for the local repository

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

1.  In Windows, click the `Start` button.
2.  Search for `Miniforge Prompt`.
3.  Change the directory to the local repository (see Step 5 in the previous section).
4.  Create the conda environment from the `conda_env_requirements.yml` file:

``` bash
conda env create -f conda_env_requirements.yml
```

5.  Activate the conda environment:

``` bash
conda activate env_AutoGluon_202502
```

6.  Launch VS Code:

``` bash
code
```

# VS Code Profile

1.  Open VS Code
2.  Type `Ctrl + Shift + P`
3.  Type `>Profiles: New Profile`
4.  Delete the profile that is called "Untitled"
5.  Click on the drop-down arrow next to the "New Profile" button
6.  Click `Import Profile`
7.  Click `Select File`
8.  Choose `Interpretable ML.code-profile`