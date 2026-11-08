"""Step boards for the two-dimensional grid DP chapter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dp231_traces_a import stories as _front
from dp231_traces_b import stories as _back

_CACHE = None


def get(problem, form):
    """Return ``(title, input line, steps, final)`` for one solution form."""
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        _CACHE.update(_front())
        _CACHE.update(_back())
    return _CACHE[(problem, form)]
