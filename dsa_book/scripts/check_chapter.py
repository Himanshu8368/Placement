"""Run every {python} cell of one or more .qmd chapters in order and report failures.

Includes (`{{< include ../problems/<key>/problem.qmd >}}`) are expanded first, exactly as Quarto
does, so registry problems are checked as part of every chapter that uses them.

Usage: python scripts/check_chapter.py books/dp/229_dp_intro.qmd [...]
"""

from __future__ import annotations

import contextlib
import re
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qmdlib import expand  # noqa: E402

CELL = re.compile(r"^```\{python\}[^\n]*\n(.*?)^```\s*$", re.S | re.M)


def check(path: Path) -> bool:
    text, included = expand(path)
    cells = CELL.findall(text)
    env: dict = {"__name__": "__chapter__"}
    start = time.perf_counter()
    # Run from the chapter's folder, as Quarto does, so `sys.path.insert(0, "../shared")` works.
    with contextlib.chdir(path.resolve().parent):
        ok = run_cells(path, text, cells, env)
    if ok:
        extra = f", {len(included)} included files" if included else ""
        print(f"OK   {path} — {len(cells)} cells{extra} in {time.perf_counter() - start:.1f}s")
    return ok


def run_cells(path: Path, text: str, cells: list[str], env: dict) -> bool:
    for i, cell in enumerate(cells, 1):
        line = text[: text.find(cell)].count("\n") + 1
        try:
            exec(compile(cell, f"{path.name}:cell{i}(line {line} after includes)", "exec"), env)
        except Exception:
            print(f"FAIL {path} cell {i} (starts at line {line} of the expanded text)")
            print("---- cell ----\n" + cell[:1500])
            traceback.print_exc()
            return False
        finally:
            import matplotlib.pyplot as plt

            plt.close("all")
    return True


if __name__ == "__main__":
    # Cells print arrows and other non-ASCII text; the Windows console defaults to cp1252.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    ok = all([check(Path(p)) for p in sys.argv[1:]])
    sys.exit(0 if ok else 1)
