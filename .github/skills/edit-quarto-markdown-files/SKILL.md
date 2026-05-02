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
