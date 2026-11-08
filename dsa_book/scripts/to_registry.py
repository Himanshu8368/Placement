"""Move the solved-problem blocks of a chapter into the problem registry.

    python scripts/to_registry.py books/dp/229_dp_intro.qmd

Every `### Problem: …` block (up to the next heading of level 1–3) goes to
books/problems/<key>/problem.qmd and is replaced in the chapter by
`{{< include ../problems/<key>/problem.qmd >}}`. Labels are kept, so cross-references still work.
The key is built from the block's `Source:` link (judge, id, name).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qmdlib import PROBLEMS  # noqa: E402

SOURCE = re.compile(r"^Source: \[(CF|LC|AC|CSES) (\S+) · ([^\]]+)\]", re.M)


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def key_for(block: str) -> str | None:
    m = SOURCE.search(block)
    if not m:
        return None
    judge, pid, name = m.groups()
    pid = pid.zfill(4) if judge == "LC" else pid.lower()
    return f"{judge.lower()}-{pid}-{slugify(name)}"[:60].rstrip("-")


def main(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").split("\n")
    out, i, moved = [], 0, []
    in_code = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and line.startswith("### Problem: "):
            j = i + 1
            code = False
            while j < len(lines):
                if lines[j].startswith("```"):
                    code = not code
                if not code and re.match(r"^#{1,3} ", lines[j]):
                    break
                j += 1
            block = "\n".join(lines[i:j]).rstrip() + "\n"
            key = key_for(block)
            if key is None:
                out.append(line)
                i += 1
                continue
            folder = PROBLEMS / key
            if folder.exists():
                # Solved already in another chapter: leave this block inline and report it, so a
                # person can replace it with the card plus what is new here.
                print(f"  DUPLICATE {key}: already solved elsewhere; left inline (replace with its card)")
                out.extend(lines[i:j])
                i = j
                continue
            folder.mkdir(parents=True)
            (folder / "problem.qmd").write_text(block, encoding="utf-8")
            out += [f"{{{{< include ../problems/{key}/problem.qmd >}}}}", ""]
            moved.append(key)
            i = j
            continue
        out.append(line)
        i += 1
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"{path}: moved {len(moved)} problems -> " + ", ".join(moved))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(Path(p))
