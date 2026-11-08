"""Snapshot helpers for the DP step boards.

Drawing lives in ``dsa_viz.story``. This module only records frames.
A frame is a dict. ``title`` is required. Everything else is optional:

- ``tree``: ``{"label", "color", "note", "kids"}`` with color
  indigo (active), green (resolved), amber (base), violet (repeated or memo)
- ``grid``: list of rows of strings. ``hi`` is ``(r, c)`` just written.
  ``marks`` maps ``(r, c)`` to a colour name (an obstacle is ``"red"``).
- ``rows``: ``[{"name", "cells", "hi"}]`` — one array per entry, ``hi`` a column index
- ``chips``: ``[{"name", "value", "color"}]`` — rolling variables
- ``lines``: plain strings
- ``eq``: the formula for this step
"""
from __future__ import annotations

import math


def show(v):
    if isinstance(v, float) and math.isinf(v):
        return "∞" if v > 0 else "−∞"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _clone(n):
    return {
        "label": n["label"], "color": n["color"], "note": n["note"],
        "kids": [_clone(k) for k in n["kids"]],
    }


def leaves(n):
    if not n or not n.get("kids"):
        return 1 if n else 0
    return sum(leaves(k) for k in n["kids"])


class CallLog:
    """Record a call tree. ``collapse=True`` drops children once a call returns,
    so a finished subtree becomes one node. Use that when the full tree is wide."""

    def __init__(self, collapse=False):
        self.collapse = collapse
        self.seen = {}
        self.stack = []
        self.root = None
        self.frames = []

    def _snap(self, title):
        if self.root is not None:
            self.frames.append({"title": title, "tree": _clone(self.root)})

    def call(self, label):
        again = label in self.seen
        node = {
            "label": label,
            "color": "violet" if again else "indigo",
            "note": "repeated" if again else "active",
            "kids": [],
        }
        if self.stack:
            self.stack[-1]["kids"].append(node)
        else:
            self.root = node
        self.stack.append(node)
        self.seen.setdefault(label, None)
        self._snap(str(label))
        return again

    def base(self, value, title=None):
        n = self.stack.pop()
        n["color"] = "amber"
        n["note"] = f"base {show(value)}"
        self.seen[n["label"]] = value
        self._snap(title or f"{n['label']} = {show(value)}")
        return value

    def ret(self, value, title=None):
        n = self.stack.pop()
        if self.collapse:
            n["kids"] = []
        n["color"] = "green"
        n["note"] = f"→ {show(value)}"
        self.seen[n["label"]] = value
        self._snap(title or f"{n['label']} = {show(value)}")
        return value

    def hit(self, label, value):
        """Memo hit. A violet leaf. Does not enter the call."""
        node = {"label": label, "color": "violet", "note": f"memo {show(value)}", "kids": []}
        self.stack[-1]["kids"].append(node)
        self._snap(f"memo {label} = {show(value)}")
        return value

    def too_wide(self, limit=7):
        return any(leaves(f["tree"]) > limit for f in self.frames)

    def steps(self, limit=8):
        frames = self.frames
        if len(frames) <= limit:
            return list(frames)
        must = {0, len(frames) - 1}
        for i, f in enumerate(frames):
            t = f["title"]
            if t.startswith("memo") or "=" in t:
                must.add(i)
        ordered = sorted(must)
        if len(ordered) > limit:
            rest = [i for i in ordered if i not in (0, frames and (len(frames) - 1))]
            # rest excludes the endpoints; keep a spread of the middle
            rest = [i for i in ordered if i != 0 and i != len(frames) - 1]
            keep = max(limit - 2, 0)
            if keep == 0 or not rest:
                ordered = [0, len(frames) - 1][:limit]
            else:
                step = len(rest) / keep
                mid = [rest[min(len(rest) - 1, int((i + 0.5) * step))] for i in range(keep)]
                ordered = sorted({0, len(frames) - 1, *mid})
        return [frames[i] for i in ordered]


def run_call(build, limit=8):
    """``build(log)`` runs one recursion and returns its answer.

    It must be safe to call twice (all of its state lives inside ``build``).
    If any frame has more than 7 leaves, it is run again with finished
    subtrees collapsed.
    """
    log = CallLog(collapse=False)
    answer = build(log)
    if log.too_wide():
        log = CallLog(collapse=True)
        answer = build(log)
    return answer, log.steps(limit)


def row_step(title, rows, eq=""):
    """rows: (name, cells, hi_index or None)."""
    return {
        "title": title,
        "rows": [{"name": n, "cells": [show(c) for c in cells], "hi": hi} for n, cells, hi in rows],
        "eq": eq,
    }


def grid_step(title, grid, hi=None, marks=None, eq=""):
    return {
        "title": title,
        "grid": [[show(c) for c in row] for row in grid],
        "hi": None if hi is None else (int(hi[0]), int(hi[1])),
        "marks": marks or {},
        "eq": eq,
    }


def chip_step(title, chips, eq=""):
    """chips: (name, value, color)."""
    return {
        "title": title,
        "chips": [{"name": n, "value": show(v), "color": c} for n, v, c in chips],
        "eq": eq,
    }


def line_step(title, lines, eq=""):
    return {"title": title, "lines": list(lines), "eq": eq}
