---
name: create-edit-agent-skill
description: Official skill generator for VS Code Agent Skills; scaffolds new skills with correct directory structure and validation, and is the upstream dependency for authoring or modifying any other skill in this repository.
---

# Create Agent Skill

This skill scaffolds high-quality Agent Skills that comply with VS Code standards and "progressive disclosure" best practices.

## Capabilities
- **Scaffold**: Creates the `.github/skills/<name>/` directory structure with a required `SKILL.md` and optional resource folders.
- **Template**: Generates a valid `SKILL.md` optimized for the agent's context window.
- **Validate**: Verifies the structure and frontmatter using the attached Python script.

## Instructions

### 1. Analyze the Request
Determine the *goal* of the new skill.
- **Name**:
   - Convert to kebab-case (e.g., "analyze-logs").
   - Prefer short, verb-led names that describe the action.
   - Keep names under 64 characters.
   - The skill folder name must exactly match `name`.
- **Description**: **CRITICAL**. This is the primary field the agent uses to decide whether to load the skill. It must clearly state what the skill does and when to use it (file types, tools, workflows, triggers).

### 2. Generate Files
Create the following structure in the workspace to support modularity. Only create the resource folders you actually need.

```text
.github/skills/<skill-name>/
├── SKILL.md           # The core instruction file
├── scripts/           # (Optional) Executable helpers for deterministic/repeated work
├── templates/         # (Optional) Response templates / standardized formats
├── references/        # (Optional) Long-form docs loaded only when needed
└── assets/            # (Optional) Files used in outputs (images, templates, boilerplate)
```

### 3. Apply the Skill Template
Use this format for the new `SKILL.md`. Put "when to use" details in `description`, because that is used for skill selection.

```markdown
---
name: <kebab-case-name>
description: <Detailed description of WHAT this skill does and WHEN to use it.>
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
`python .github/skills/create-edit-agent-skill/scripts/validate.py .github/skills/<skill-name>`

## Examples

### Example 1: PDF Reader Skill
**Input**: "Create a skill for reading PDF files."
**Output**:
- Directory: `.github/skills/pdf-reader`
- Structure: Includes `scripts/extract_pdf.py` and `templates/summary_format.md`
- Description: "Extracts text from PDF files using a Python script and summarizes the content."
- Instructions: "Run `./scripts/extract_pdf.py` to parse the file, then display results using `./templates/summary_format.md`."

## Best Practices
- **Concise is key**: Keep `SKILL.md` focused on essential workflow. Offload details into `references/` and keep those files discoverable from `SKILL.md`.
- **Progressive disclosure**: Prefer a small `SKILL.md` + optional `references/` files; load detailed docs only when needed.
- **Avoid clutter**: Do not add extra documentation files (README, changelog, etc.) that don't directly help the agent execute tasks.
- **Relative paths**: Refer to resources using relative paths (e.g., `./scripts/myscript.py`).
- **Description is key**: If the agent isn't using your skill, the `description` is likely too vague—make triggers explicit.
