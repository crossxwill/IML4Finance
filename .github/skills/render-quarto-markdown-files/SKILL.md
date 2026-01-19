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
