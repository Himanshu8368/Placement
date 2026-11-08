"""Structural lint for written chapters — catches the mistakes we keep repeating.

Run on the numeric chapters under books/<folder>/NNN_<slug>.qmd (the list Quarto
also builds). For each chapter it flags, outside code fences and HTML comments:

  ERROR  a local Windows path (e.g. D:\\…) in rendered text — dead link on the web
  ERROR  an unbalanced fenced div (::: opened but never closed, or a stray :::)
  ERROR  a '::: {.div}' opened on the line after text — Pandoc reads it as a
         paragraph continuation and prints the fence literally
  WARN   a chapter with no "## Further reading" section
  WARN   a "## Further reading" bullet that is not a clickable [link](http… url)
  WARN   a chapter longer than about 120 pages (400 words a page, includes expanded): split it

Exit code is 1 only when there are ERRORS, so a genuine mistake blocks the build
while the (expected, non-blocking) warnings still print. See STYLE.md.

Usage: python scripts/check_style.py [file.qmd ...]   # defaults to every chapter
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAPTER_GLOB = "*-[0-9][0-9][0-9]_*.qmd"  # matched per folder below
DRIVE_PATH = re.compile(r"[A-Za-z]:\\")
OPEN_FENCE = re.compile(r"^:::\s+\S")   # ::: {.callout} / ::: {#fig-...}
CLOSE_FENCE = re.compile(r"^:::\s*$")    # bare :::
CODE_FENCE = re.compile(r"^(```|~~~)")
WORDS_PER_PAGE = 400          # prose
CODE_LINES_PER_PAGE = 50      # listings
PAGE_LIMIT = 120


def _visible_lines(text: str):
    """Yield (lineno, line) for lines that render (not in a ``` fence or <!-- -->)."""
    in_code = in_comment = False
    for i, line in enumerate(text.splitlines(), 1):
        if CODE_FENCE.match(line.strip()):
            in_code = not in_code
            continue
        if in_code:
            continue
        # track HTML comments (can span lines)
        if in_comment:
            if "-->" in line:
                in_comment = False
            continue
        if "<!--" in line and "-->" not in line:
            in_comment = True
            continue
        yield i, line


def check(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    text = path.read_text(encoding="utf-8")
    visible = list(_visible_lines(text))

    # 1. Local Windows paths in rendered text.
    for lineno, line in visible:
        if DRIVE_PATH.search(line):
            errors.append(f"{path.name}:{lineno}: local path in rendered text: {line.strip()[:80]}")

    # 1b. An unescaped % inside $math$ is a LaTeX comment: the PDF build fails (HTML hides it).
    for lineno, line in visible:
        for m in re.finditer(r"(?<![\\$])\$([^$`]+)\$", re.sub(r"`[^`]*`", "", line)):
            if re.search(r"(?<!\\)%", m[1]):
                errors.append(f"{path.name}:{lineno}: unescaped % in math ${m[1][:40]}$ breaks the PDF; "
                              f"write \\% or put the code in backticks")

    # 2. Fenced-div balance (::: opened vs closed, tracking depth).
    depth = 0
    for lineno, line in visible:
        if OPEN_FENCE.match(line):
            depth += 1
        elif CLOSE_FENCE.match(line):
            depth -= 1
            if depth < 0:
                errors.append(f"{path.name}:{lineno}: stray ':::' close (no matching open)")
                depth = 0
    if depth:
        errors.append(f"{path.name}: {depth} fenced div(s) opened with ':::' but never closed")

    # 3. A div must open on its own block: keep a blank line above '::: {...',
    #    otherwise the preceding paragraph swallows the fence as literal text.
    raw = text.splitlines()
    for lineno, line in visible:
        if OPEN_FENCE.match(line):
            prev = raw[lineno - 2] if lineno > 1 else ""
            if prev.strip() and not CODE_FENCE.match(prev.strip()):
                errors.append(f"{path.name}:{lineno}: ':::' div needs a blank line above it "
                              f"(previous line: {prev.strip()[:60]!r})")

    # 3b. `V.show(draw)` indented under `def draw` is part of the function body, never runs, and the
    #     figure silently disappears. It must sit at the cell's top level.
    in_py = False
    for lineno, line in enumerate(raw, 1):
        if line.startswith("```{python}"):
            in_py = True
        elif line.startswith("```"):
            in_py = False
        elif in_py and re.match(r"^\s+V\.(show|plot)\(draw\b", line):
            errors.append(f"{path.name}:{lineno}: V.show(draw) is indented inside the function, so the "
                          f"figure never renders; dedent it to the cell's top level")

    # 4 & 5. Further reading present, and every bullet hyperlinked.
    fr_start = next((i for i, (_, l) in enumerate(visible) if l.strip() == "## Further reading"), None)
    name = path.name
    if path.parent.parent.name == "problems":      # registry fragment: lives inside a chapter
        return errors, warnings
    # 6. Length: about 400 words per printed page, includes expanded. Above PAGE_LIMIT pages the
    #    chapter should be split (the book splits chapters, it never compresses them).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qmdlib import expand
    full = expand(path)[0]
    code = re.findall(r"^```.*?^```", full, flags=re.S | re.M)
    prose = re.sub(r"^```.*?^```", " ", full, flags=re.S | re.M)
    pages = len(re.findall(r"\S+", prose)) / WORDS_PER_PAGE + sum(c.count("\n") for c in code) / CODE_LINES_PER_PAGE
    if pages > PAGE_LIMIT:
        warnings.append(f"{name}: about {pages:.0f} pages (includes expanded); "
                        f"above {PAGE_LIMIT}, split it into two chapters in outline.yml")
    if fr_start is None:
        warnings.append(f"{name}: no '## Further reading' section")
    else:
        bullets = []
        for _, line in visible[fr_start + 1:]:
            if line.startswith("## "):
                break
            if re.match(r"^[-*]\s+\S", line):
                bullets.append(line)
        if not bullets:
            warnings.append(f"{name}: '## Further reading' is empty")
        for b in bullets:
            if not re.search(r"\]\(\s*https?://", b):
                warnings.append(f"{name}: further-reading bullet has no hyperlink: {b[:70]}")
    return errors, warnings


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        chapters = [Path(p) for p in argv[1:]]
    else:
        chapters = sorted(ROOT.glob(f"books/*/[0-9][0-9][0-9]_*.qmd"))
        # Registry problems are linted on their own too (chapters only include them).
        chapters += sorted(ROOT.glob("books/problems/*/*.qmd"))
    all_errors: list[str] = []
    all_warnings: list[str] = []
    for ch in chapters:
        e, w = check(ch)
        all_errors += e
        all_warnings += w
    for w in all_warnings:
        print(f"WARN  {w}")
    for e in all_errors:
        print(f"ERROR {e}")
    print(f"check_style: {len(chapters)} chapters — {len(all_errors)} error(s), {len(all_warnings)} warning(s)")
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
