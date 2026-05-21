---
name: edit-quarto-markdown-files
description: Edit Quarto (.qmd) and Markdown (.md) files with proper nested code block handling to prevent parser errors and truncated output; standalone utility with no skill dependencies.
---

## When to Use
- Editing `.qmd` (Quarto) files
- Editing `.md` (Markdown) files
- Creating or modifying documentation with code examples
- Any file containing nested markdown code blocks

## Critical: Nested Markdown Code Blocks
When editing markdown files that contain code blocks, you must use **four backticks** (````) for the outer fence to properly nest three-backtick code examples inside.

### Why This Matters
If you use three backticks for both the outer fence and inner code examples, the markdown parser will incorrectly close the outer block at the first inner closing fence, causing:
- Truncated or malformed output
- Lost content after the first code example
- Broken formatting in rendered documents

### Correct Pattern

Use four backticks for the outer fence when showing code block examples:

````markdown
## Example Section

Here is how to write Python code:

```python
def hello():
    print("Hello, world!")
```

And here is R code:

```r
hello <- function() {
  print("Hello, world!")
}
```
````

### Incorrect Pattern (Causes Truncation)

Do NOT use three backticks for both outer and inner fences:

```markdown
## Example Section

Here is Python code:

```python  <!-- This closes the outer block! -->
def hello():
    print("Hello, world!")
```
```

## Quarto-Specific Guidelines

### YAML Front Matter
Quarto files start with YAML front matter between `---` delimiters:

````qmd
---
title: "Document Title"
format: html
execute:
  echo: true
---

# Content starts here
````

### Executable Code Chunks
Quarto code chunks can have execution options:

````qmd
```{python}
#| label: fig-example
#| fig-cap: "Example figure"
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
```
````

When editing executable chunks, remember that Quarto will run them exactly as written. Small Python defects inside one chunk can fail a full render late in the document.

### Cross-References
Use `@fig-`, `@tbl-`, `@sec-` prefixes for cross-references:

````qmd
See @fig-example for the plot.
See @tbl-results for the data.
See @sec-methods for methodology.
````

## Best Practices

1. **Always use four backticks** when your content contains code block examples
2. **Preserve YAML front matter** - don't accidentally modify document metadata
3. **Maintain consistent indentation** in nested lists and code chunks
4. **Use proper Quarto chunk options** with `#|` syntax for execution control
5. **Test rendering** after edits to ensure code blocks display correctly
6. **When fixing one repeated code chunk, check sibling chunks with the same structure** so a later copy does not fail on the next render pass
7. **Avoid shadowing imported helpers inside notebook-style code** - for example, do not reuse names like `display`, `list`, or `dict` for loop variables if those callables are used later in the same chunk
8. **For executable Python chunks, do a cheap local syntax check when practical** before starting a long full-document render

## Code-Chunk Safety Checks

When editing `.qmd` files with executable Python chunks:

1. Verify that any opened function call, list, dict, or parenthesized expression is fully closed before the closing code fence.
2. If a chunk was copied from a nearby section, compare it against the sibling chunk to catch truncated lines or missed renames.
3. Watch for local variable names that overwrite imported helpers such as `display` from `IPython.display`.
4. If the document is long-running, validate the edited snippet with a cheap syntax compile or a very narrow check before rerendering the entire file.

These checks are especially important in teaching labs and notebooks where the same modeling/calibration block is repeated across multiple strategies.

## File Extensions Reference

| Extension | Description |
|-----------|-------------|
| `.qmd` | Quarto markdown document |
| `.md` | Standard markdown |
| `.Rmd` | R Markdown (similar to Quarto) |
| `.ipynb` | Jupyter notebook (can be edited with Quarto) |

## Example Edit Task

When asked to add a code example to a markdown file, structure your edit like this:

````markdown
## New Section

Here's how to load data in Python:

```python
import pandas as pd
df = pd.read_csv("data.csv")
print(df.head())
```

Output:
```
   col1  col2
0     1     a
1     2     b
```
````

This ensures all nested code blocks render correctly without truncation.
