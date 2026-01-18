---
name: create-edit-agent-skill
description: Official skill generator for GitHub Copilot Agent Skills; scaffolds new skills with correct directory structure and validation, and is the upstream dependency for authoring or modifying any other skill in this repository.
---

# Create Agent Skill

This skill scaffolds high-quality Agent Skills that comply with VS Code, GitHub Copilot standards, and "Progressive Loading" best practices.

## Capabilities
- **Scaffold**: Creates the `.github/skills/<name>/` directory structure including `scripts/` and `templates/`.
- **Template**: Generates a valid `SKILL.md` optimized for the Agent's context window.
- **Validate**: Verifies the structure using the attached Python script.

## Instructions

### 1. Analyze the Request
Determine the *goal* of the new skill.
- **Name**: Convert to kebab-case (e.g., "analyze-logs").
- **Description**: **CRITICAL**. This is the *only* field the Agent reads initially. It must be highly descriptive and action-oriented so the Agent selects it automatically without user prompting.

### 2. Generate Files
Create the following structure in the workspace to support modularity:

```text
.github/skills/<skill-name>/
├── SKILL.md           # The core instruction file
├── templates/         # (Optional) Markdown templates for standardized responses
└── scripts/           # (Optional) Python/Node.js scripts for complex logic
```

### 3. Apply the Skill Template
Use this format for the new `SKILL.md`. Note the use of relative paths for scripts and templates:

```markdown
---
name: <kebab-case-name>
description: <Detailed description of WHAT this skill does. Agent uses this to decide whether to load the full skill.>
---

# <Human Readable Title>

## Purpose
<2-3 sentences explaining the problem this skill solves>

## Usage
- <Trigger phrase 1>
- <Trigger phrase 2>

## Instructions
1. <Step 1: e.g., Run the analysis script>
   - Run: `./scripts/analyze.py`
2. <Step 2: e.g., Format the output>
   - Respond using the format defined in: `./templates/response.md`
```

### 4. Verify Structure
After creating the files, run the validation script to ensure compliance:
`python .github/skills/create-agent-skill/scripts/validate.py .github/skills/<skill-name>`

## Examples

### Example 1: PDF Reader Skill
**Input**: "Create a skill for reading PDF files."
**Output**:
- Directory: `.github/skills/pdf-reader`
- Structure: Includes `scripts/extract_pdf.py` and `templates/summary_format.md`
- Description: "Extracts text from PDF files using a Python script and summarizes the content."
- Instructions: "Run `./scripts/extract_pdf.py` to parse the file, then display results using `./templates/summary_format.md`."

## Best Practices
- **Progressive Loading**: Keep the `SKILL.md` file concise. Offload complex logic to files in `./scripts/` and long text formats to `./templates/`. The Agent reads the `SKILL.md` *after* selection, so a smaller file saves tokens and reduces latency.
- **Relative Paths**: Always refer to scripts and templates using relative paths (e.g., `./scripts/myscript.py`).
- **Description is Key**: If the Agent isn't using your skill, the description is likely too vague. Make it keyword-rich.