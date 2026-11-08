"""Run every {python} cell of one or more .qmd chapters in order and report failures.

Usage: python scripts/check_chapter.py books/part01_math/01_linear_algebra.qmd [...]
"""

from __future__ import annotations

import re
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

CELL = re.compile(r"^```\{python\}[^\n]*\n(.*?)^```\s*$", re.S | re.M)


def check(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    cells = CELL.findall(text)
    env: dict = {"__name__": "__chapter__"}
    start = time.perf_counter()
    for i, cell in enumerate(cells, 1):
        line = text[: text.find(cell)].count("\n") + 1
        try:
            exec(compile(cell, f"{path.name}:cell{i}(line {line})", "exec"), env)
        except Exception:
            print(f"FAIL {path} cell {i} (starts at line {line})")
            traceback.print_exc()
            return False
        finally:
            import matplotlib.pyplot as plt

            plt.close("all")
    print(f"OK   {path} — {len(cells)} cells in {time.perf_counter() - start:.1f}s")
    return True


if __name__ == "__main__":
    ok = all([check(Path(p)) for p in sys.argv[1:]])
    sys.exit(0 if ok else 1)
