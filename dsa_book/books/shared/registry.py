"""Load the trace module that belongs to one registry problem.

A problem whose figures need helper code keeps it next to the problem, in
books/problems/<key>/traces.py, so the problem folder is self-contained and moves as one piece.
Inside problem.qmd (or solution.qmd):

    ```{python}
    #| include: false
    import registry
    T = registry.traces("lc-0070-climbing-stairs")
    ```

gen_book.py hashes that file with the chapter, so editing it re-runs every chapter that includes the problem.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROBLEMS = Path(__file__).resolve().parent.parent / "problems"
_cache: dict = {}


def traces(key: str):
    if key not in _cache:
        path = PROBLEMS / key / "traces.py"
        if not path.exists():
            raise FileNotFoundError(f"no traces.py for problem {key!r} (expected {path})")
        spec = importlib.util.spec_from_file_location(f"traces_{key.replace('-', '_')}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _cache[key] = mod
    return _cache[key]
