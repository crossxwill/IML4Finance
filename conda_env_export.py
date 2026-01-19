import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    print("Missing dependency: PyYAML. Install it in your environment and try again.")
    sys.exit(1)


class _IndentDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):  # type: ignore[override]
        return super().increase_indent(flow, False)


def _export_env(env_name: str, from_history: bool, output_path: Path) -> None:
    conda_exe = (
        os.environ.get("CONDA_EXE")
        or shutil.which("conda.exe")
        or shutil.which("conda")
        or shutil.which("conda.bat")
    )
    if not conda_exe:
        raise FileNotFoundError(
            "Unable to locate conda executable. Activate conda first or ensure it is on PATH."
        )

    cmd = [conda_exe, "env", "export", "--no-build", "-n", env_name]
    if from_history:
        cmd.append("--from-history")

    if conda_exe.lower().endswith(".bat"):
        cmd_str = " ".join(cmd)
        result = subprocess.run(cmd_str, capture_output=True, text=True, check=True, shell=True)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    output_path.write_text(result.stdout, encoding="utf-8")


def _extract_lock_versions(lock_data: dict) -> dict[str, str]:
    versions: dict[str, str] = {}
    for dep in lock_data.get("dependencies", []):
        if isinstance(dep, str):
            parts = dep.split("=")
            if len(parts) >= 2:
                name = parts[0]
                version = parts[1]
                versions[name] = version
    return versions


def _extract_pip_list(data: dict) -> list[str]:
    for dep in data.get("dependencies", []):
        if isinstance(dep, dict) and "pip" in dep:
            return dep["pip"] or []
    return []


def _update_base_env(base_data: dict, lock_versions: dict[str, str], lock_pip: list[str]) -> None:
    new_dependencies: list = []
    pip_replaced = False

    for dep in base_data.get("dependencies", []):
        if isinstance(dep, str):
            name = dep.split("=")[0]
            if name in lock_versions:
                new_dependencies.append(f"{name}={lock_versions[name]}")
            else:
                new_dependencies.append(dep)
        elif isinstance(dep, dict) and "pip" in dep:
            new_dependencies.append({"pip": lock_pip})
            pip_replaced = True
        else:
            new_dependencies.append(dep)

    if not pip_replaced and lock_pip:
        new_dependencies.append({"pip": lock_pip})

    base_data["dependencies"] = new_dependencies


def main() -> None:
    env_name = input("Conda environment name: ").strip()
    if not env_name:
        print("Environment name is required.")
        sys.exit(1)

    base_path = Path.cwd() / f"conda_{env_name}.yml"
    lock_path = Path.cwd() / f"conda_{env_name}_lock.yml"

    _export_env(env_name, from_history=True, output_path=base_path)
    _export_env(env_name, from_history=False, output_path=lock_path)

    lock_data = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    base_data = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}

    lock_versions = _extract_lock_versions(lock_data)
    lock_pip = _extract_pip_list(lock_data)

    _update_base_env(base_data, lock_versions, lock_pip)

    base_path.write_text(
        yaml.dump(
            base_data,
            Dumper=_IndentDumper,
            sort_keys=False,
            default_flow_style=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Updated {base_path.name} using {lock_path.name}.")


if __name__ == "__main__":
    main()
