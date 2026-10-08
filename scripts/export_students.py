#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["nbstripout>=0.8,<1"]
# ///
"""Export the current working tree as a student repository.

Run from any directory with:
    uv run --script /path/to/hands-on/scripts/export_students.py

Prompts for the student repository name (default: <repository-name>-students).
Use --repo-name NAME to skip the prompt. The name sets the default sibling
folder and the editable-install path in the exported setup script.
Use --destination PATH to choose another location. Add --overwrite to rebuild
an existing export (replacing local files, preserving its .git directory).
No source Git history is copied or initialized.
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".uv-cache",
    ".aws", ".codex", ".agents", "tmp", "node_modules",
}
EXCLUDED_FILES = {
    Path("M03_Retrieval_Augmented_Generation/M03_03_RAG_Conflict.ipynb"),
    Path("M03_Retrieval_Augmented_Generation/M03_04_RAG_Visual.ipynb"),
}


def include(path: Path) -> bool:
    """Keep scaffolding and data, excluding local state and instructor material."""
    if path in EXCLUDED_FILES:
        return False
    if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in path.parts):
        return False
    if path.name == ".DS_Store" or path.suffix in {".pyc", ".pyo"}:
        return False
    if (path.name == ".env" or path.name.startswith(".env.")) and path.name not in {
        ".env.example", ".env.sample", ".env.template",
    }:
        return False
    if path.parts[:2] == ("data", "generated"):
        return False
    if path.parts[0] == "M06_AI_Code_Development" and path.stem == "TASK_BRIEF":
        return False
    return path != Path("scripts/export_students.py")


def project_files() -> list[Path]:
    # Export current files, including uncommitted edits and new scaffolding.
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    )
    paths = sorted({Path(p.decode("utf-8")) for p in result.stdout.split(b"\0") if p})
    selected = []
    for path in paths:
        if not include(path):
            continue
        source = ROOT / path
        if source.is_symlink():
            raise ValueError(f"Refusing to export a symbolic link: {path}")
        if source.is_file():
            selected.append(path)
    return selected


def update_student_references(folder: Path) -> None:
    module = folder / "M06_AI_Code_Development"
    readme = module / "README.md"
    if readme.exists():
        text = "".join(
            line for line in readme.read_text().splitlines(keepends=True)
            if not line.startswith("**Instructor reference:**")
        )
        text = text.replace(" Do not provide the detailed task brief.", "")
        readme.write_text(text)
    for path in module.rglob("*.py"):
        text = path.read_text()
        text = text.replace("See TASK_BRIEF.md and tests/test_agent.py", "See tests/test_agent.py")
        text = text.replace("CLI contract in TASK_BRIEF.md", "CLI contract in tests/test_cli.py")
        path.write_text(text)


def export(destination: Path, *, repo_name: str, overwrite: bool = False) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", repo_name):
        raise ValueError("Repository name must start with a letter or digit and contain only letters, digits, dots, underscores, or hyphens.")
    if destination == ROOT or ROOT in destination.parents or destination in ROOT.parents:
        raise ValueError("The destination must be outside the source repository and its ancestors.")
    if destination.is_symlink():
        raise ValueError(f"Refusing to replace a symbolic link: {destination}")
    if destination.exists():
        if not overwrite:
            raise ValueError(f"Destination already exists: {destination}. Use --overwrite to rebuild it.")
        if not destination.is_dir():
            raise ValueError(f"Destination is not a directory: {destination}")
    if not destination.parent.is_dir():
        raise ValueError(f"Destination parent does not exist: {destination.parent}")
    # Check the dependency before copying anything.
    subprocess.run([sys.executable, "-m", "nbstripout", "--version"], check=True,
                   stdout=subprocess.DEVNULL)
    files = project_files()
    notebooks = [p for p in files if p.suffix == ".ipynb"]
    removed = 0
    with tempfile.TemporaryDirectory(prefix=".student-export-", dir=destination.parent) as tmp:
        staging = Path(tmp) / "repository"
        staging.mkdir()
        for relative in files:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        for relative in notebooks:
            target = staging / relative
            original = json.loads(target.read_text())
            expected = [c for c in original["cells"]
                        if "solution" not in c.get("metadata", {}).get("tags", [])]
            subprocess.run(
                [sys.executable, "-m", "nbstripout", "--drop-tagged-cells", "solution",
                 "--keep-id", str(target)], check=True,
            )
            result = json.loads(target.read_text())
            if len(result["cells"]) != len(expected):
                raise ValueError(f"Unexpected cell count after stripping: {relative}")
            for before, after in zip(expected, result["cells"]):
                if (before["source"] != after["source"]
                        or before["cell_type"] != after["cell_type"]
                        or "solution" in after.get("metadata", {}).get("tags", [])
                        or after.get("outputs") or after.get("execution_count") is not None):
                    raise ValueError(f"Notebook verification failed: {relative}")
            removed += len(original["cells"]) - len(expected)
        update_student_references(staging)
        setup = staging / "init.d/setup_project.sh"
        if setup.exists():
            text, count = re.subn(
                r"^uv pip install -e .+$", f"uv pip install -e {repo_name}",
                setup.read_text(), flags=re.MULTILINE,
            )
            if count != 1:
                raise ValueError("Expected exactly one editable-install command in init.d/setup_project.sh.")
            setup.write_text(text)
        if destination.exists() and not overwrite:
            raise ValueError(f"Destination appeared during export: {destination}")
        backup = Path(tmp) / "previous-export"
        if destination.exists():
            destination.rename(backup)
        try:
            if (backup / ".git").exists():
                (backup / ".git").rename(staging / ".git")
            staging.rename(destination)
        except BaseException:
            if (staging / ".git").exists() and backup.exists():
                (staging / ".git").rename(backup / ".git")
            if backup.exists():
                backup.rename(destination)
            raise
    print(f"Created student repository: {destination}")
    print(f"Copied {len(files)} files and stripped {removed} solution cells from {len(notebooks)} notebooks.")
    print("Notebook outputs cleared; M06 TASK_BRIEF and Git history excluded.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--destination", type=Path, help="Override the destination folder.")
    parser.add_argument("--repo-name", help="Student repository name; skips the interactive prompt.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild an existing destination, replacing local files and preserving .git.")
    args = parser.parse_args()
    try:
        default_name = args.destination.name if args.destination else f"{ROOT.name}-exercises"
        repo_name = args.repo_name
        if repo_name is None:
            repo_name = input(f"Student repository name [{default_name}]: ").strip() or default_name
        destination = (args.destination or ROOT.parent / repo_name).expanduser()
        if destination.is_symlink():
            raise ValueError(f"Refusing to replace a symbolic link: {destination}")
        export(destination.resolve(), repo_name=repo_name, overwrite=args.overwrite)
    except (ValueError, OSError, EOFError, subprocess.CalledProcessError) as error:
        print(f"Export failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
