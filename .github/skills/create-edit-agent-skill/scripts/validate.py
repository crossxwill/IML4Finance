import os
import sys
import re

def validate_skill(skill_path):
    print(f"🔍 Validating Skill at: {skill_path}")

    # 1. Validate Directory Existence
    if not os.path.isdir(skill_path):
        fail(f"Directory not found: {skill_path}")
    
    # 2. Validate SKILL.md Existence
    skill_md_path = os.path.join(skill_path, "SKILL.md")
    if not os.path.exists(skill_md_path):
        fail("Missing 'SKILL.md'. This file is required.")

    # 3. Read and Parse SKILL.md
    with open(skill_md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 4. Validate Frontmatter (Supports --- and ***)
    # Regex looks for content between the first two sets of --- or ***
    frontmatter_match = re.match(r'^(\-\-\-|\*\*\*)(.*?)(\1)', content, re.DOTALL)
    
    if not frontmatter_match:
        fail("Missing or malformed YAML frontmatter. Must start and end with '---' or '***'.")
    
    frontmatter_text = frontmatter_match.group(2)
    
    # Simple parsing of name and description
    name_match = re.search(r'name:\s*(.+)', frontmatter_text)
    desc_match = re.search(r'description:\s*(.+)', frontmatter_text)

    if not name_match:
        fail("Frontmatter missing 'name' field.")
    if not desc_match:
        fail("Frontmatter missing 'description' field.")

    skill_name = name_match.group(1).strip()
    description = desc_match.group(1).strip()

    # 5. Validate Kebab-Case Name
    if not re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', skill_name):
        fail(f"Skill name '{skill_name}' must be kebab-case (lowercase, hyphens only).")

    # 6. Validate Description Quality (Heuristic)
    if len(description) < 20:
        warn("Description is very short. A detailed description is critical for the Agent to know when to pick this skill.")

    # 7. Check for Modular Structure (Best Practice)
    scripts_dir = os.path.join(skill_path, "scripts")
    templates_dir = os.path.join(skill_path, "templates")
    
    has_scripts = os.path.isdir(scripts_dir)
    has_templates = os.path.isdir(templates_dir)

    # Scan content for references to scripts/templates
    if "./scripts/" in content and not has_scripts:
        fail("SKILL.md references './scripts/', but the 'scripts' directory is missing.")
    
    if "./templates/" in content and not has_templates:
        fail("SKILL.md references './templates/', but the 'templates' directory is missing.")

    # 8. Success Message
    print("\n✅ Skill Validation Passed!")
    print(f"   - Name: {skill_name}")
    print(f"   - Modules: {'Scripts' if has_scripts else 'None'}, {'Templates' if has_templates else 'None'}")

def fail(message):
    print(f"\n❌ ERROR: {message}")
    sys.exit(1)

def warn(message):
    print(f"\n⚠️  WARNING: {message}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate.py <path_to_skill_directory>")
        sys.exit(1)
    
    validate_skill(sys.argv[1])