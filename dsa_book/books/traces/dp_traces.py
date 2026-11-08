"""Step boards for every solved problem in the DP introduction chapter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dp_traces_grid import stories as _grid
from dp_traces_house import stories as _house
from dp_traces_kadane import stories as _kadane
from dp_traces_linear import stories as _linear
from dp_traces_state import stories as _state

_CACHE = None


def get(problem, form):
    """Return ``(title, input line, steps, final)`` for one solution form."""
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        for part in (_linear(), _house(), _grid(), _state(), _kadane()):
            _CACHE.update(part)
    return _CACHE[(problem, form)]
