"""Helpers shared by the build scripts: Quarto-style include expansion and problem-registry paths.

Quarto resolves `{{< include path >}}` relative to the *chapter* being rendered, also inside an
included file. Every chapter lives one folder below `books/`, so registry files are always written
as `../problems/<key>/problem.qmd` and `../problems/<key>/solution.qmd`.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ROOT / "books"
PROBLEMS = BOOKS / "problems"
INCLUDE = re.compile(r"^\{\{<\s*include\s+(\S+)\s*>\}\}\s*$", re.M)


def expand(path: Path, base: Path | None = None, seen: list[Path] | None = None) -> tuple[str, list[Path]]:
    """Return the text of `path` with includes expanded (recursively), and the list of included files."""
    base = base or path.resolve().parent
    seen = [] if seen is None else seen
    text = path.read_text(encoding="utf-8")

    def sub(m: re.Match) -> str:
        target = (base / m.group(1)).resolve()
        if not target.exists():
            raise FileNotFoundError(f"{path}: include not found: {m.group(1)}")
        seen.append(target)
        return expand(target, base, seen)[0]

    return INCLUDE.sub(sub, text), seen


def problem_key_of(path: Path) -> str | None:
    """`books/problems/<key>/problem.qmd` -> `<key>`."""
    try:
        rel = path.resolve().relative_to(PROBLEMS)
    except ValueError:
        return None
    return rel.parts[0] if len(rel.parts) > 1 else None


IMPORT = re.compile(r"^\s*(?:import|from)\s+(\w+)", re.M)


def deps_hash(chapter: Path) -> str:
    """Hash of everything a chapter's output depends on: the chapter, every file it includes, and every
    local module it imports from books/shared, books/traces or a problem folder (traces.py)."""
    import hashlib

    text, included = expand(chapter)
    local = {p.stem: p for d in ("shared", "traces") for p in (BOOKS / d).glob("*.py")}
    extra = [inc.parent / "traces.py" for inc in included if (inc.parent / "traces.py").exists()]
    deps = [chapter, *included, *extra] + [local[m] for m in sorted(set(IMPORT.findall(text))) if m in local]
    h = hashlib.sha1()
    for d in deps:
        h.update(d.read_bytes())
    return h.hexdigest()


def chapter_files() -> list[Path]:
    """Every chapter (numbered .qmd one folder below books/), excluding the registry and the appendix."""
    return sorted(p for p in BOOKS.glob("*/[0-9][0-9][0-9]_*.qmd")
                  if p.parent.name not in ("problems", "appendix", "_build", "_freeze"))
