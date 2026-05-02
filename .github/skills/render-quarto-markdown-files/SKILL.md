---
name: quarto-qmd-cli-rendering
description: Render or build or compile Quarto .qmd files from the command line.
---

## Goal

When asked to “render”, “build”, “preview”, or “compile” a Quarto `.qmd`, respond with `quarto` CLI commands that are copy-pastable and include brief, practical notes.

Include a short note that the terminal will show per-cell progress lines like:

```
Cell 1/3 '{cell-label}'.....Done
Cell 2/3 '{cell-label}'.....Done
Cell 3/3 '{cell-label}'.....Done

Output created: {file-name}
```

Some cells finish quickly while others can take a long time; the agent should be patient and wait for completion. For example, the following terminal indicates that cell 2 is still running:

```
Cell 1/3 '{cell-label}'.....Done
Cell 2/3 '{cell-label}'.....
```

## Assumptions

- Quarto CLI is installed and available as `quarto`.
- The user can run commands from a terminal in the relevant directory, or will provide an explicit path.
- The target output format is either defined in YAML, or the user will specify it.

## Core rules

- Single-file render: use `quarto render <file.qmd>`.
- Project/directory render: use `quarto render` or `quarto render <dir>` (named project directory).
- Prefer explicit success signaling in scripts, but match the user's shell:
   - PowerShell: append `; if ($LASTEXITCODE -eq 0) { echo "Render finished" }`
   - bash/sh: append `&& echo "Render finished"`
- If the user says “render sequentially” or “low RAM”, render one file at a time (do not suggest parallel execution).

## Render a single `.qmd`

Minimal pattern:

```bash
quarto render <file.qmd>
```

Optional explicit success signal (choose ONE depending on shell):

```powershell
quarto render <file.qmd> ; if ($LASTEXITCODE -eq 0) { echo "Render finished" }
```

```bash
quarto render <file.qmd> && echo "Render finished"
```

## Sequential rendering (low RAM)

Use sequential rendering when the user requests:
- “One at a time.”
- “No parallelism.”
- “Low RAM” or “don’t run everything at once.”
- “Render each qmd and stop on error.”

Instructions:
1. Render exactly one file:
   - PowerShell: `quarto render <file.qmd> ; if ($LASTEXITCODE -eq 0) { echo "Render finished" }`
   - bash/sh: `quarto render <file.qmd> && echo "Render finished"`
2. Wait until the command exits (i.e., when the terminal shows "Render finished") before proceeding to the next file.
3. If it fails, fix the current `.qmd`, then re-run the same command until it succeeds.
4. Continue to the next `.qmd` only after the current one succeeds.
5. The command succeeds when there are no error messages in the terminal and you see `Render finished`.

## Skill scope

Use this skill when:
- The user mentions `.qmd`, Quarto, “render/build/compile/preview”, CI, Makefiles, or scripting.

Do not use this skill when:
- The user explicitly wants GUI-only steps (RStudio/VS Code) with no CLI commands.
## Troubleshooting common rendering errors

### Error: "Unable to locate an installed version of Python 3"

Quarto looks for `python3` on the PATH, but conda environments only provide `python` (not `python3`). Fix by setting the `QUARTO_PYTHON` environment variable to the full path of the Python executable **before** running `quarto render`.

First, find the Python path:
```powershell
conda activate <env_name>; python -c "import sys; print(sys.executable)"
```

Then use it when rendering:
```powershell
conda activate <env_name>; $env:QUARTO_PYTHON = "<full-python-path>"; quarto render <file.qmd>; if ($LASTEXITCODE -eq 0) { echo "Render finished" }
```

Example:
```powershell
conda activate env_iml4finance_2026; $env:QUARTO_PYTHON = "C:\Users\chiuw\miniforge3\envs\env_iml4finance_2026\python.exe"; quarto render Labs/Lab_01.qmd; if ($LASTEXITCODE -eq 0) { echo "Render finished" }
```

### Error: "Unknown project cache version" / "Assertion failed" (Sass/revealjs)

This occurs when Quarto's internal cache is stale or was created by a different Quarto version. The error often manifests during Sass compilation for revealjs themes.

Fix: Clear all Quarto cache directories recursively before rendering:
```powershell
conda activate <env_name>; Get-ChildItem -Recurse -Directory -Filter "_quarto*" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue; Get-ChildItem -Recurse -Directory -Filter ".quarto*" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue; Get-ChildItem -Recurse -Directory -Filter "_site*" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue; echo "All caches cleared"
```

Then retry the render with `--no-cache`:
```powershell
conda activate <env_name>; $env:QUARTO_PYTHON = "<full-python-path>"; quarto render <file.qmd> --no-cache; if ($LASTEXITCODE -eq 0) { echo "Render finished" }
```

### Error: "PermissionDenied: Access is denied. (os error 5)" during cleanup

Harmless cleanup error. Use `-ErrorAction SilentlyContinue` on `Remove-Item` commands to suppress it. Does not affect render output.

### General pre-render checklist

Before rendering any `.qmd` in this project, always:
1. Activate the conda environment: `conda activate env_iml4finance_2026`
2. Set `$env:QUARTO_PYTHON` to the conda env's Python executable
3. Clear stale Quarto caches if previous renders failed
4. Use `--no-cache` if cache issues persist
5. Append the success signal: `; if ($LASTEXITCODE -eq 0) { echo "Render finished" }`

### Render time expectations

- **revealjs slides** (Lectures): Fast — typically under 30 seconds for slides with no executable code cells
- **HTML labs with AutoGluon training**: Slow — can take 5–30+ minutes depending on `time_limit` settings. Cells that train models (e.g., `TabularPredictor.fit()`) are the bottleneck
- **HTML labs without model training**: Moderate — typically 1–5 minutes for EDA and plotting cells

Always use `mode='async'` when running render commands and poll with `get_terminal_output` until the success signal appears.