from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

EXCLUDE_DIRS = {
    ".git", ".github", ".idea", ".vscode", "node_modules", "vendor", "venv",
    ".venv", "__pycache__", "build", "dist", "out", "target", "bin", "obj",
    ".gradle", ".mvn", "gradle", "test", "tests", "src/test", "testing",
    "generated", ".next", "coverage",
}


@dataclass
class SourceFile:
    path: str          # absolute
    rel_path: str      # relative to repo root
    ext: str
    text: str
    n_lines: int


@dataclass
class LoadedRepo:
    root: str
    files: list[SourceFile]
    readme: str | None


def _is_excluded(rel_path: str) -> bool:
    parts = {p.lower() for p in Path(rel_path).parts}
    return bool(parts & EXCLUDE_DIRS)


def _clone(url: str) -> str:
    dest = tempfile.mkdtemp(prefix="cba_repo_")
    subprocess.run(
        ["git", "clone", "--depth", "1", url, dest],
        check=True, capture_output=True, text=True,
    )
    return dest


def _find_readme(root: str) -> str | None:
    for name in ("README.md", "README.rst", "README.txt", "README"):
        p = Path(root) / name
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8", errors="replace")[:20_000]
            except OSError:
                return None
    return None


def load_repo(source: str, extensions: tuple[str, ...], max_files: int = 0) -> LoadedRepo:
    """Load a repo from a local path or a git URL."""
    root = _clone(source) if source.startswith(("http://", "https://", "git@")) else source
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"not a directory: {root}")

    exts = {e.lower() for e in extensions}
    files: list[SourceFile] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs in place so we don't descend into them
        dirnames[:] = [d for d in dirnames if d.lower() not in EXCLUDE_DIRS]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in exts:
                continue
            abs_path = os.path.join(dirpath, fn)
            rel = os.path.relpath(abs_path, root)
            if _is_excluded(rel):
                continue
            try:
                text = Path(abs_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            files.append(
                SourceFile(
                    path=abs_path, rel_path=rel, ext=ext,
                    text=text, n_lines=text.count("\n") + 1,
                )
            )

    files.sort(key=lambda f: f.rel_path)
    if max_files > 0:
        files = files[:max_files]
    return LoadedRepo(root=root, files=files, readme=_find_readme(root))
