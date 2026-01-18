---
name: quarto-qmd-cli-rendering
description: Render or compile Quarto .qmd files from the command line (single files and projects), including output formats and parameterized renders.
---

## Goal

When asked to “render”, “build”, “preview”, or “compile” a Quarto `.qmd`, use the `quarto` CLI and provide copy-pastable commands plus brief explanations.

## Assumptions

- Quarto CLI is installed and available as `quarto` in PATH.
- The target `.qmd` file path is known, and the command is run from a terminal in the relevant directory (or uses an explicit path).

## Core rule

Always put the target input file (for example `report.qmd`) as the **first** command line argument to `quarto render`.

## Render a single `.qmd`

Use this pattern:

```bash
quarto render path/to/report.qmd
```

To specify output format explicitly:

```bash
quarto render path/to/report.qmd --to html
quarto render path/to/report.qmd --to pdf
quarto render path/to/report.qmd --to docx
```

Quarto’s own tutorial shows `--to html` and `--to docx` examples using this exact `quarto render <file> --to <format>` pattern.

## Render a project or directory

If the user asks to “render everything” in a folder or project, run `quarto render` on the directory, or `cd` into it and run `quarto render` with no file argument:

```bash
quarto render subdir
# or
cd subdir
quarto render
```

Quarto will render all valid Quarto input files (including `.qmd`) in that directory when rendering a directory/project.

## Parameterized renders (optional)

If the user mentions parameters (for example “run with alpha=0.2”), use Quarto render parameters:

- Inline parameters with `-P`:

```bash
quarto render report.qmd -P alpha:0.2 -P ratio:0.3
```

- Parameters from a YAML file with `--execute-params`:

```bash
quarto render report.qmd --execute-params params.yml
```

These patterns are documented in community Quarto rendering guides and match Quarto CLI usage.

## What to output in responses

When generating an answer for the user, include:

- The exact command(s) to run (minimal, copy-pasteable).
- If multiple files/formats are mentioned, show a small set of commands (2–4 max) and ask a clarifying question if the target format is unclear.
- A reminder that `<file>.qmd` must come first in `quarto render <file>.qmd ...`.

## Clarifying questions Copilot should ask

Ask one question if needed:

- “Which output format do you want (html, pdf, docx)?”
- “Is this a single file render, or a Quarto project render?”
- “Should code execute during render, or is this a render of existing outputs?”

## Skill scope

Use this skill when:
- The user references `.qmd`, “Quarto”, “render”, “build”, “preview”, or “compile” from CLI.
- The user wants commands for CI, Makefiles, or scripts (provide the same CLI commands).

Do not use this skill when:
- The user explicitly wants GUI instructions only (RStudio/VS Code UI) without CLI commands.
