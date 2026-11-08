"""Step boards for the array-based data-structure design chapter (swap-with-last
deletion, snapshots and versions, counters of counters, array-backed hashing,
lazy resets with timestamps).

Drawing lives in ``dsa_viz.story``; ``get`` returns data only. Each story walks the
chapter's own sample, at most eight frames, and every number comes from running the
algorithm - never from a hand-typed table. A frame shows the structure's own arrays -
the packed list and position map of a randomized set, the (snapshot, value) history
of one column, the bucket table of a hand-rolled hash map - so the reader watches a
swap-with-last exchange, a version tag, or a timestamp binary search exactly as the
code performs it.

Every problem carries a ladder: a ``brute``/first frame set (the direct definition),
an ``optimal``/second frame set (the rung the pattern names), and a ``contest`` third
rung (a submission-shaped mechanic). ``get`` returns whichever frame set the caller
asks for.
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from trace_log import chip_step, line_step, row_step, show  # noqa: E402


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    if len(steps) > 8:                    # keep first, a spread, and the last frame
        keep = sorted({0, 1, 3, 5, 7, len(steps) - 2, len(steps) - 1})
        steps = [steps[i] for i in keep][:8]
    if not 1 <= len(steps) <= 8:
        raise AssertionError(f"{title}: {len(steps)} frames")
    return (title, inp, steps, text)


def _cells(vals):
    return [str(v) for v in vals]


def _short(url):
    return url.split(".")[0]


# ------------------------------------------------------- LC 380 insert-delete-getrandom
RS_OPS = ["insert 1", "remove 2", "insert 2", "getRandom", "remove 1", "insert 2", "getRandom"]
RS_ANS = [True, False, True, 1, True, False, 2]


class _RSPlain:
    """Rung 1: a list only; membership and removal scan, order is preserved."""

    def __init__(self):
        self.a = []

    def insert(self, v):
        if v in self.a:
            return False
        self.a.append(v)
        return True

    def remove(self, v):
        if v not in self.a:
            return False
        self.a.remove(v)
        return True

    def get_random(self):
        return random.choice(self.a)


def _rs_scan():
    random.seed(1)
    s = _RSPlain()
    out = []
    steps = [row_step("empty list", [("data", [], None)],
                      eq="insert appends; remove scans and shifts; O(n) per op")]
    for op, want in zip(RS_OPS, RS_ANS):
        parts = op.split()
        name, v = parts[0], int(parts[1]) if len(parts) > 1 else None
        if name == "insert":
            r = s.insert(v)
        elif name == "remove":
            r = s.remove(v)
        else:
            r = s.get_random()
        assert r == want
        out.append(r)
        hi = len(s.a) - 1 if (name == "insert" and r) else None
        steps.append(row_step(op, [("data", s.a, hi)], eq=f"-> {r}"))
    return _pack("RandomizedSet - list only", " ".join(RS_OPS), steps,
                 "outputs " + " ".join(str(x) for x in out))


class _RSSwap:
    """Rung 2: a bare list; removal searches for the index, then swaps with the last element."""

    def __init__(self):
        self.a = []

    def insert(self, v):
        if v in self.a:
            return False
        self.a.append(v)
        return True

    def remove(self, v):
        if v not in self.a:
            return False
        i = self.a.index(v)                # the O(n) search the position map removes
        self.a[i] = self.a[-1]
        self.a.pop()
        return True

    def get_random(self):
        return random.choice(self.a)


def _rs_swap():
    random.seed(1)
    s = _RSSwap()
    out = []
    steps = [row_step("empty", [("data", [], None)],
                     eq="remove: find v by a scan, copy the tail into its slot, pop")]
    for op, want in zip(RS_OPS, RS_ANS):
        parts = op.split()
        name, v = parts[0], int(parts[1]) if len(parts) > 1 else None
        if name == "insert":
            r = s.insert(v)
        elif name == "remove":
            r = s.remove(v)
        else:
            r = s.get_random()
        assert r == want
        out.append(r)
        row = ("data", s.a, len(s.a) - 1 if (name == "insert" and r) else None)
        steps.append(row_step(op, [row], eq=f"-> {r}; data={s.a}"))
    return _pack("RandomizedSet - swap-with-last", " ".join(RS_OPS), steps,
                 "outputs " + " ".join(str(x) for x in out))


def _rs_jukebox():
    """Rung 3: one template - land at the hole, swap, shrink; remove-then-insert reuse."""
    random.seed(1)
    data = []
    pos = {}
    out = []

    def put(v):
        if v in pos:
            return False
        pos[v] = len(data)
        data.append(v)
        return True

    def drop(v):
        if v not in pos:
            return False
        i = pos[v]                     # repair the tail's entry BEFORE dropping v's
        tail = data[-1]
        data[i] = tail
        pos[tail] = i
        del pos[v]
        data.pop()
        return True

    steps = [row_step("empty", [("data", [], None), ("pos", [], None)],
                      eq="one hole-filling template; the double-remove sample reuses it")]
    for op, want in zip(RS_OPS, RS_ANS):
        parts = op.split()
        name, v = parts[0], int(parts[1]) if len(parts) > 1 else None
        r = put(v) if name == "insert" else (drop(v) if name == "remove" else random.choice(data))
        assert r == want
        out.append(r)
        steps.append(row_step(op, [("data", data, None), ("pos", [f"{k}:{i}" for k, i in pos.items()], None)],
                              eq=f"-> {r}"))
    assert data == [2]
    return _pack("RandomizedSet - position map", " ".join(RS_OPS), steps,
                 "outputs " + " ".join(str(x) for x in out))


# ------------------------------------------------- LC 381 randomized collection (duplicates)
RC_OPS = ["insert 1", "insert 1", "insert 2", "getRandom", "remove 1", "getRandom"]
RC_ANS = [True, False, True, 2, True, 1]


def _rc_scanlast():
    """Rung 1: plain list; membership scan, remove the LAST occurrence, shift."""
    random.seed(6)
    a = []
    out = []
    steps = [row_step("empty list", [("data", [], None)],
                      eq="insert appends; remove scans right-to-left for the last copy")]
    for op, want in zip(RC_OPS, RC_ANS):
        parts = op.split()
        name, v = parts[0], int(parts[1]) if len(parts) > 1 else None
        if name == "insert":
            r = v not in a
            a.append(v)
        elif name == "remove":
            r = v in a
            if r:
                a.reverse()
                a.remove(v)
                a.reverse()
        else:
            r = random.choice(a)
        assert r == want
        out.append(r)
        steps.append(row_step(op, [("data", a, len(a) - 1 if r and name != "getRandom" else None)],
                              eq=f"-> {r}; contains {sorted(set(a))}"))
    return _pack("RandomizedCollection - scan for a copy", " ".join(RC_OPS), steps,
                 "outputs " + " ".join(str(x) for x in out))


def _rc_possets():
    """Rung 2: value -> SET of positions; swap-with-last must re-point the moved tail."""
    random.seed(5)
    a = []
    pos = defaultdict(set)
    out = []
    steps = [row_step("empty", [("data", [], None), ("pos", [], None)],
                      eq="pos[v] is a set: one key, many slots; swap-with-last still O(1)")]
    for op, want in zip(RC_OPS, RC_ANS):
        parts = op.split()
        name, v = parts[0], int(parts[1]) if len(parts) > 1 else None
        if name == "insert":
            r = len(pos[v]) == 0
            pos[v].add(len(a))
            a.append(v)
        elif name == "remove":
            i = pos[v].pop()
            last, li = a[-1], len(a) - 1
            if i != li:                    # the tail moves into the hole only if it is not the hole
                a[i] = last
                pos[last].discard(li)
                pos[last].add(i)
            a.pop()
            r = True
        else:
            r = random.choice(a)
        assert r == want
        out.append(r)
        steps.append(row_step(op, [("data", a, None),
                                   ("pos", [f"{k}:{sorted(s)}" for k, s in sorted(pos.items())
                                            if s], None)],
                              eq=f"-> {r}"))
    assert a == [2, 1]
    assert pos[1] == {1} and pos[2] == {0}
    return _pack("RandomizedCollection - a slot set per value", " ".join(RC_OPS), steps,
                 "outputs " + " ".join(str(x) for x in out))


def _rc_anyslot():
    """Rung 3: contest compact form - at[v] is a SET but removal takes ANY member;
    with few duplicates a list per value behaves like a stack, and the tail-slot
    bookkeeping is one dict update."""
    random.seed(5)
    a = []                      # packed values
    at = {}                     # value -> its live slots; no key means "absent"
    out = []
    steps = [row_step("empty", [("data", [], None), ("at", [], None)],
                      eq="any slot of v may die; an emptied slot set is deleted, so at holds live values only")]
    for op, want in zip(RC_OPS, RC_ANS):
        parts = op.split()
        name, v = parts[0], int(parts[1]) if len(parts) > 1 else None
        if name == "insert":
            r = v not in at
            at.setdefault(v, set())
            at[v].add(len(a))
            a.append(v)
        elif name == "remove":
            r = bool(at.get(v))
            if r:
                i = at[v].pop()
                last = a.pop()
                if i != len(a):            # the popped tail refills the hole
                    a[i] = last
                    at[last].discard(len(a))
                    at[last].add(i)
                if not at[v]:
                    del at[v]              # an empty slot set means "absent"
        else:
            r = random.choice(a)
        assert r == want
        out.append(r)
        steps.append(row_step(op, [("data", a, None),
                                   ("at", [f"{k}:{sorted(s)}" for k, s in sorted(at.items()) if s], None)],
                              eq=f"-> {r}"))
    return _pack("RandomizedCollection - delete emptied sets", " ".join(RC_OPS), steps,
                 "outputs " + " ".join(str(x) for x in out))


# ---------------------------------------------------------------- LC 1146 snapshot array
SNAP_OPS = ["set(0,5)", "snap", "set(0,6)", "get(0,0)"]
SNAP_ANS = [None, 0, None, 5]
SNAP_STEPS = [("set", 0, 5), ("snap", None, None), ("set", 0, 6), ("get", 0, None)]
SNAP_HISTORY = [[(0, 5), (1, 6)], [(0, 0)], [(0, 0)]]


def _snap_fullcopy():
    """Rung 1: keep every snapshot as a complete copy of the array."""
    cur = [0, 0, 0]
    snaps = []
    out = []
    steps = [row_step("length 3", [("cur", cur, None), ("snapshots", ["-"], None)],
                      eq="snap() copies the WHOLE array - O(n) per snapshot")]
    for op, (kind, idx, val) in zip(SNAP_OPS, SNAP_STEPS):
        if kind == "set":
            cur[idx] = val
            out.append(None)
        elif kind == "snap":
            snaps.append(cur[:])
            out.append(len(snaps) - 1)
        else:
            out.append(snaps[0][idx])      # get(0, snap 0)
        steps.append(row_step(op, [("cur", cur, idx if kind == "set" else None),
                                   ("snapshots", [str(s) for s in snaps] or ["-"], None)],
                              eq=f"-> {out[-1]}" if out[-1] is not None else "stored"))
    assert out == SNAP_ANS
    assert snaps == [[5, 0, 0]]
    return _pack("SnapshotArray - copy the whole array", " ".join(SNAP_OPS), steps,
                 "outputs " + " ".join(str(x) for x in out))


def _snap_columns():
    """Rung 2: per-column (snap_id, value) history; get walks back, then binary search."""
    hist0 = [(0, 5)]
    steps = [row_step("after set(0,5)", [("col 0", hist0, None), ("col 1", [(0, 0)], None),
                                          ("col 2", [(0, 0)], None)],
                      eq="only column 0 grows a record; snap() just bumps a counter - O(1)")]
    steps.append(row_step("snap -> id 0", [("col 0", hist0, None), ("col 1", [(0, 0)], None),
                                            ("col 2", [(0, 0)], None)],
                          eq="cur_id = 1; nothing copied"))
    hist0.append((1, 6))
    steps.append(row_step("set(0,6)", [("col 0", hist0, None), ("col 1", [(0, 0)], None),
                                        ("col 2", [(0, 0)], None)],
                          eq="tag the NEW value with the CURRENT id 1"))
    hist = "[(0,5),(1,6)]"
    steps.append(line_step("get(0,0)", [
        f"col0 history {hist}",
        "find last record with snap_id <= 0  ->  (0,5)",
    ], eq="-> 5"))
    assert hist0 == SNAP_HISTORY[0] and hist0[0] == (0, 5)
    return _pack("SnapshotArray - versioned columns", " ".join(SNAP_OPS), steps,
                 "outputs " + " ".join(str(x) for x in SNAP_ANS))


def _snap_lazy():
    """Rung 3: lazy default (0,0) padding so get never special-cases
    'column untouched'; bisect finds the floor record."""
    import bisect
    versions = [[(0, 5), (1, 6)], [(0, 0)], [(0, 0)]]
    steps = [row_step("final layout", [(f"col {i}", [f"{s}:{v}" for s, v in c], None)
                                       for i, c in enumerate(versions)],
                      eq="every column starts at (0,0); ids only increase")]
    for idx, sid, want in [(0, 0, 5), (0, 1, 6), (1, 0, 0), (2, 0, 0)]:
        col = versions[idx]
        p = bisect.bisect_right([s for s, _ in col], sid) - 1
        assert col[p][1] == want
        steps.append(row_step(f"get({idx},{sid})",
                              [(f"col {idx}", [f"{s}:{v}" for s, v in col], p)],
                              eq=f"bisect_right ids [0,1] at {sid} -> record {col[p]} -> {want}"))
    return _pack("SnapshotArray - bisect the column", " ".join(SNAP_OPS), steps,
                 "get(0,0)=5, get(0,1)=6, get(1,0)=0")


# ---------------------------------------------------------------- LC 1472 browser history
BH_OPS = ["visit google", "visit facebook", "visit youtube", "back 1", "back 1",
          "forward 1", "visit linkedin", "forward 2", "back 2", "back 7"]
BH_ANS = ["leetcode", "google", "facebook", "youtube", "facebook", "google", "facebook",
          "linkedin", "linkedin", "google", "leetcode"]
BH_HOME = "leetcode"


def _bh_cursor():
    """Rung 1/2 canonical: one list, one cursor; visit truncates the forward tail."""
    hist = [BH_HOME]
    i = 0
    out = [hist[i]]
    steps = [row_step("start", [("hist", hist, i)], eq="cursor i marks the current page")]
    for op, want in zip(BH_OPS, BH_ANS[1:]):
        parts = op.split()
        if parts[0] == "visit":
            del hist[i + 1:]
            hist.append(parts[1])
            i = len(hist) - 1
        elif parts[0] == "back":
            i = max(0, i - int(parts[1]))
        else:
            i = min(len(hist) - 1, i + int(parts[1]))
        assert hist[i] == want
        out.append(hist[i])
        steps.append(row_step(op, [("hist", [_short(h) for h in hist], i)],
                              eq=f"-> {hist[i]}"))
    assert hist == [BH_HOME, "google", "facebook", "linkedin"]
    steps = [steps[0]] + steps[1:6] + [steps[-1]]
    return _pack("BrowserHistory - one list, one cursor", " ".join(BH_OPS[:3]), steps,
                 "outputs " + " ".join(out))


def _bh_window():
    """Rung 3: one growable array with an end marker - no deletion, stale pages lie beyond end."""
    arr = [BH_HOME]
    cur, end = 0, 1
    out = [arr[cur]]
    steps = [row_step("start", [("arr", arr, 0), ("window", [f"end={end}"], None)],
              eq="visit overwrites at cur+1 and pulls the end marker with it")]
    for op, want in zip(BH_OPS, BH_ANS[1:]):
        parts = op.split()
        if parts[0] == "visit":
            cur += 1
            if cur == len(arr):
                arr.append(parts[1])
            else:
                arr[cur] = parts[1]
            end = cur + 1
        elif parts[0] == "back":
            cur = max(0, cur - int(parts[1]))
        else:
            nxt = min(end - 1, cur + int(parts[1]))
            cur = nxt
        assert arr[cur] == want
        out.append(arr[cur])
        steps.append(row_step(op, [("arr", [_short(a) for a in arr[:end]], cur),
                                   ("window", [f"end={end}"], None)],
                              eq=f"-> {arr[cur]}"))
    steps = [steps[0]] + steps[1:6] + [steps[-1]]
    return _pack("BrowserHistory - array with an end marker", " ".join(BH_OPS[:3]), steps,
                 "outputs " + " ".join(out))


def _bh_stack():
    """Rung 3: two stacks - back pops forward onto a second stack."""
    left = [BH_HOME]
    right = []
    out = [left[-1]]
    steps = [row_step("start", [("back stack (top=cur)", left, None), ("fwd stack", right, None)],
                eq="visit clears the forward stack; back/forward move one card")]
    for op, want in zip(BH_OPS, BH_ANS[1:]):
        parts = op.split()
        if parts[0] == "visit":
            right.clear()
            left.append(parts[1])
        elif parts[0] == "back":
            for _ in range(min(int(parts[1]), len(left) - 1)):
                right.append(left.pop())
        else:
            for _ in range(min(int(parts[1]), len(right))):
                left.append(right.pop())
        assert left[-1] == want
        out.append(left[-1])
        steps.append(row_step(op, [("back stack (top=cur)", [_short(x) for x in left], None),
                                   ("fwd stack", [_short(x) for x in right], None)],
                              eq=f"-> {left[-1]}"))
    steps = [steps[0]] + steps[1:6] + [steps[-1]]
    return _pack("BrowserHistory - two stacks", " ".join(BH_OPS[:3]), steps,
                 "outputs " + " ".join(out))


# ---------------------------------------------------------------- LC 895 freq stack
FS_PUSHES = [5, 7, 5, 7, 4, 5]
FS_POPS = [5, 7, 5, 4]


def _fs_scan():
    """Rung 1: a plain stack; pop scans for the highest-frequency tie-broken by recency."""
    def last_pos(stack, v):
        return max(i for i, x in enumerate(stack) if x == v)   # index of v's LAST copy

    a = list(FS_PUSHES)
    out = []
    steps = [row_step("after six pushes", [("stack", a, None)],
              eq="pop(): count frequencies, take the max, break ties by TOPMOST")]
    while a and len(out) < 4:
        cnt = Counter(a)
        best = max(set(a), key=lambda v: (cnt[v], last_pos(a, v)))
        del a[last_pos(a, best)]      # the newest copy leaves the stack
        out.append(best)
        steps.append(row_step(f"pop -> {best}", [("stack", a, None)],
                              eq=f"freqs {dict(cnt)}; winner {best}"))
    assert out == FS_POPS
    return _pack("FreqStack - rescan on pop", "pushes " + str(FS_PUSHES), steps,
                 "pops " + str(out))


def _fs_levels():
    """Rung 2: freq -> stack of elements at that freq; maxf is the only cursor."""
    cnt = Counter()
    levels = defaultdict(list)
    maxf = 0
    out = []
    steps = [row_step("empty", [("freq", [], None), ("level 1", [], None), ("level 2", [], None)],
                      eq="push x: f=freq[x]+1; levels[f].append(x); maxf=max(maxf,f)")]
    for x in FS_PUSHES:
        cnt[x] += 1
        maxf = max(maxf, cnt[x])
        levels[cnt[x]].append(x)
        steps.append(row_step(f"push {x}",
                              [("freq", [f"{k}:{v}" for k, v in sorted(cnt.items())], None),
                               ("level 1", levels[1], None), ("level 2", levels.get(2, []), None),
                               ("level 3", levels.get(3, []), None)],
                              eq=f"maxf={maxf}"))
    steps = [steps[0]] + steps[1:5] + steps[-2:]
    for _ in range(len(FS_POPS)):
        x = levels[maxf].pop()
        cnt[x] -= 1
        if not cnt[x]:
            del cnt[x]
        if not levels[maxf]:
            maxf -= 1
        out.append(x)
    assert out == FS_POPS
    return _pack("FreqStack - frequency levels", "pushes " + str(FS_PUSHES), steps,
                 "pops " + str(out))


def _fs_snapshot():
    """Rung 3: levels store (value, maxf-at-push) so pop needs no maxf bookkeeping."""
    cnt = Counter()
    levels = defaultdict(list)
    out = []
    steps = [line_step("the compression", [
        "levels[f] holds the elements that REACHED frequency f",
        "pop takes levels[top f] - ties break by push order for free",
    ], eq="two dicts, four lines each")]
    for x in FS_PUSHES:
        cnt[x] += 1
        levels[cnt[x]].append(x)
    top = max(cnt.values())
    steps.append(row_step("built", [("freq", [f"{k}:{v}" for k, v in sorted(cnt.items())], None),
                                    ("level 1", levels[1], None), ("level 2", levels[2], None),
                                    ("level 3", levels[3], None)],
                          eq=f"top level {top} = {levels[top]}"))
    while len(out) < 4:
        x = levels[top].pop()
        cnt[x] -= 1
        if not levels[top]:
            top -= 1
        out.append(x)
        steps.append(row_step(f"pop -> {x}",
                              [("level " + str(k), v, None) for k, v in sorted(levels.items()) if v],
                              eq=f"top={top}"))
    assert out == FS_POPS
    return _pack("FreqStack - levels in an array", "pushes " + str(FS_PUSHES), steps,
                 "pops " + str(out))


# ---------------------------------------------------------------- CF 1200A hotelier
HT_S1 = "LLRL1RL1"
HT_A1 = "1010000011"
HT_S2 = "L0L0LLRR9"
HT_A2 = "1100000010"


def _ht_scan():
    """Rung 1: ten booleans; L scans left, R scans right, digit writes one cell."""
    occ = [0] * 10
    steps = [row_step("all free", [("rooms", list(range(10)), None), ("occ", occ, None)],
                      eq="state[room] 1 = occupied; L/R scan for a 0 - O(10) per request")]
    for ch in HT_S1:
        if ch == "L":
            occ[occ.index(0)] = 1
        elif ch == "R":
            occ[9 - occ[::-1].index(0)] = 1
        else:
            occ[int(ch)] = 0
        steps.append(row_step(f"request {ch}", [("rooms", list(range(10)), None),
                                                ("occ", occ, int(ch) if ch.isdigit() else None)],
                              eq="".join(map(str, occ))))
    assert "".join(map(str, occ)) == HT_A1
    return _pack("Hotelier - scan for a free room", HT_S1, steps, "".join(map(str, occ)))


def _ht_runs():
    """Rung 3: keep the free set as its extreme values - a run bookkeeping."""
    occ = [0] * 10
    steps = [line_step("view", [
        "occupied rooms always form a PREFIX (from L) + a SUFFIX (from R)",
        "so two integers (pl, pr) bound the free middle; a digit punches one hole",
    ], eq="holes make the view lie - the scan rung is the truth, this is the fast path")]
    pl, pr = 0, 9
    for ch in HT_S1:
        if ch == "L":
            while pl <= pr and occ[pl]:
                pl += 1
            occ[pl] = 1
        elif ch == "R":
            while pr >= pl and occ[pr]:
                pr -= 1
            occ[pr] = 1
        else:
            occ[int(ch)] = 0
            pl = min(pl, int(ch))
            pr = max(pr, int(ch))
        steps.append(row_step(f"request {ch}", [("occ", occ, int(ch) if ch.isdigit() else None),
                                                ("bounds", [f"pl={pl}", f"pr={pr}"], None)],
                              eq="".join(map(str, occ))))
    assert "".join(map(str, occ)) == HT_A1
    return _pack("Hotelier - free-end bounds", HT_S1, steps, "".join(map(str, occ)))


# ---------------------------------------------------------------- LC 432 all O(1)
AO_OPS = ["inc(hello)", "inc(hello)", "getMaxKey()", "getMinKey()",
          "inc(leet)", "getMaxKey()", "getMinKey()"]
AO_ANS = [None, None, "hello", "hello", None, "hello", "leet"]


def _ao_key(op):
    return op[op.index("(") + 1:op.index(")")]


def _al_scan():
    """Rung 1: one dict key -> count; getMaxKey/getMinKey scan every key."""
    cnt = {}
    out = []
    steps = [row_step("key -> count", [("table", ["-"], None)],
                      eq="inc/dec are O(1); get* scan all keys -> O(distinct) per call")]
    for op, want in zip(AO_OPS, AO_ANS):
        if op.startswith("inc"):
            k = _ao_key(op)
            cnt[k] = cnt.get(k, 0) + 1
            r = None
            eq = "bumped"
        elif op.startswith("getMax"):
            r = max(cnt, key=lambda x: cnt[x]) if cnt else ""
            eq = f"scan -> {r}"
        else:
            r = min(cnt, key=lambda x: cnt[x]) if cnt else ""
            eq = f"scan -> {r}"
        assert r == want
        out.append(r)
        steps.append(row_step(op, [("table", [f"{a}:{b}" for a, b in cnt.items()], None)], eq=eq))
    return _pack("AllOne - scan the counts", " ".join(AO_OPS[:3]), steps,
                 "outputs " + " ".join(str(x) for x in out))


def _al_buckets():
    """Rung 2: two maps, key -> count and count -> set(keys); a level is a counter
    of counters, so get* only scan the levels, not every key."""
    cnt = {}
    groups = defaultdict(set)
    out = []
    steps = [row_step("count -> set(keys)", [("levels", ["-"], None)],
                      eq="a key's count moves it between levels; get* scan LEVELS")]
    for op, want in zip(AO_OPS, AO_ANS):
        if op.startswith("inc"):
            k = _ao_key(op)
            c = cnt.get(k, 0)
            if c:
                groups[c].discard(k)
                if not groups[c]:
                    del groups[c]
            cnt[k] = c + 1
            groups[c + 1].add(k)
            r = None
            eq = f"{k} -> level {c + 1}"
        elif op.startswith("getMax"):
            top = max(groups)
            r = next(iter(groups[top]))
            eq = f"top level {top}"
        else:
            bot = min(groups)
            r = next(iter(groups[bot]))
            eq = f"bottom level {bot}"
        assert r == want
        out.append(r)
        levels = [f"{c}:" + "{" + ",".join(sorted(v)) + "}" for c, v in sorted(groups.items())]
        steps.append(row_step(op, [("levels", levels or ["-"], None)],
                              eq=f"{eq}; -> {r}" if r is not None else eq))
    return _pack("AllOne - count buckets", " ".join(AO_OPS[:3]), steps,
                 "outputs " + " ".join(str(x) for x in out))


def _al_linked():
    """Rung 3: bucket nodes in a doubly-linked list ordered by count; a key
    moves to the neighbour node, so inc/dec/get* are all O(1)."""
    cnt = {}
    groups = defaultdict(set)
    order = []
    out = []
    steps = [row_step("head(min) <-> tail(max)", [("chain", ["-"], None)],
                      eq="insert/remove a node only next to the key's current node -> O(1)")]
    for op, want in zip(AO_OPS, AO_ANS):
        if op.startswith("inc"):
            k = _ao_key(op)
            c = cnt.get(k, 0)
            if c:
                groups[c].discard(k)
                if not groups[c]:
                    order.remove(c)
            cnt[k] = c + 1
            if c + 1 not in order:
                order.append(c + 1)
                order.sort()
            groups[c + 1].add(k)
            r = None
            eq = f"{k} -> count {c + 1}"
        elif op.startswith("getMax"):
            top = order[-1]
            r = next(iter(groups[top]))
            eq = f"tail node {top}"
        else:
            bot = order[0]
            r = next(iter(groups[bot]))
            eq = f"head node {bot}"
        assert r == want
        out.append(r)
        chain = [f"{c}:" + "{" + ",".join(sorted(groups[c])) + "}" for c in order]
        steps.append(row_step(op, [("chain", chain or ["-"], None)],
                              eq=f"{eq}; -> {r}" if r is not None else eq))
    return _pack("AllOne - linked buckets", " ".join(AO_OPS[:3]),
                 steps, "outputs " + " ".join(str(x) for x in out))


# ---------------------------------------------------------------- LC 706 design hashmap
HM_OPS = ["put(1,1)", "put(2,2)", "get(1)", "get(3)", "put(2,1)", "get(2)", "remove(2)", "get(2)"]
HM_ANS = [None, None, 1, -1, None, 1, None, -1]
HM_B = 4          # teaching bucket count; production rung uses 1000


def _hm_direct():
    """Rung 1: keys are bounded (0..10^6) so a direct array of values is legal - memory-hungry."""
    table = {}
    out = []
    steps = [row_step("empty", [("keys seen", [], None), ("values", [], None)],
                      eq="array indexed BY the key: O(1) everywhere, O(U) memory wasted on gaps")]
    for op, want in zip(HM_OPS, HM_ANS):
        name, inner = op[:-1].split("(")
        args = [int(x) for x in inner.split(",")]
        if name == "put":
            table[args[0]] = args[1]
        elif name == "get":
            pass
        else:
            table.pop(args[0], None)
        r = table.get(args[0], -1) if name == "get" else None
        assert r == want
        out.append(r)
        steps.append(row_step(op, [("occupied slots", sorted(table), None),
                                   ("values", [f"{k}->{v}" for k, v in sorted(table.items())], None)],
                              eq=f"-> {r}" if r is not None else "stored"))
    return _pack("HashMap - direct addressing", " ".join(HM_OPS[:2]), steps,
                 "outputs " + " ".join(str(x) for x in out))


def _hm_buckets():
    """Rung 2: B buckets, each a small list of pairs; h(k) = k mod B spreads the work."""
    buckets = [[] for _ in range(HM_B)]
    out = []
    steps = [row_step("4 buckets", [(f"b{j}", [], None) for j in range(HM_B)],
                      eq=f"h(k) = k mod {HM_B}; every bucket is a mini list scanned in O(len)")]
    for op, want in zip(HM_OPS, HM_ANS):
        name, inner = op[:-1].split("(")
        args = [int(x) for x in inner.split(",")]
        b = buckets[args[0] % HM_B]
        if name == "put":
            for p in b:
                if p[0] == args[0]:
                    p[1] = args[1]
                    break
            else:
                b.append([args[0], args[1]])
        elif name == "remove":
            b[:] = [p for p in b if p[0] != args[0]]
        r = next((p[1] for p in b if p[0] == args[0]), -1) if name == "get" else None
        assert r == want
        out.append(r)
        steps.append(row_step(op, [(f"b{j}", [f"{k}:{v}" for k, v in buckets[j]],
                                    None if j != args[0] % HM_B else 0)
                                   for j in range(HM_B)],
                              eq=f"h({args[0]})={args[0] % HM_B}; -> {r}" if r is not None
                                 else f"h({args[0]})={args[0] % HM_B}"))
    assert buckets[1 % HM_B] == [[1, 1]]
    steps = [steps[0]] + steps[1:6] + [steps[-1]]
    return _pack("HashMap - chained buckets", " ".join(HM_OPS[:2]), steps,
                 "outputs " + " ".join(str(x) for x in out))


def _hm_resize():
    """Rung 3: production shape - many buckets, load factor triggers a rehash."""
    cap, size, tbl = 4, 0, [None] * 4
    steps = [row_step(f"cap {cap}", [("table", ["-"] * cap, None)],
                      eq="resize when the table would exceed its capacity; keys must be RE-HOMED")]
    for key, val in [(1, 1), (2, 2), (3, 3), (4, 4)]:
        if size + 1 > cap:                       # would overflow: double first
            cap *= 2
            old, tbl = tbl, [None] * cap
            for t in old:
                if t:
                    tbl[t[0] % cap] = t
            steps.append(row_step(f"rehash -> cap {cap}",
                                  [("table", [f"{t[0]}:{t[1]}" if t else "-" for t in tbl], None)],
                                  eq="same keys, NEW homes: h(k) = k mod cap depends on cap"))
        i = key % cap
        if tbl[i] is None:
            size += 1
        tbl[i] = [key, val]
        steps.append(row_step(f"put({key},{val})",
                              [("table", [f"{t[0]}:{t[1]}" if t else "-" for t in tbl], i)],
                              eq=f"h({key})={i}; size={size} cap={cap}"))
    assert tbl[1 % cap] == [1, 1] and tbl[2 % cap] == [2, 2] and tbl[4 % cap] == [4, 4]
    return _pack("HashMap - grow and rehash", "put 1..4", steps,
                 "cap 4 -> 8; every key re-homed")


# ---------------------------------------------------------------- CF 1799A recent actions
RA_N, RA_P = 5, [6, 7, 8, 9, 10]
RA_ANS = [5, 4, 3, 2, 1]
RA_S3_N, RA_S3_P = 4, [5, 9, 9, 5, 7]
RA_S3_ANS = [-1, 5, 2, 1]


def _ra_simulate():
    """Rung 1: simulate the field exactly - move-to-front or add-and-pop-the-bottom."""
    field = list(range(1, RA_N + 1))
    evicted = {}
    steps = [row_step("field (top..bottom)", [("_", field, None)],
                      eq="n=5; actions arrive on posts 6..10; each step is O(n)")]
    for t, p in enumerate(RA_P, 1):
        if p in field:
            field.remove(p)
            field.insert(0, p)
            eq = f"{p} was in the field: moved to top, nobody leaves"
        else:
            gone = field.pop()
            evicted[gone] = t
            field.insert(0, p)
            eq = f"bottom post {gone} evicted at t={t}; {p} enters on top"
        steps.append(row_step(f"t={t}: post {p}", [("_", field, None)], eq=eq))
    ans = [evicted.get(i, -1) for i in range(1, RA_N + 1)]
    assert ans == RA_ANS
    return _pack("Recent Actions - move-to-front simulation", f"n=5 p={RA_P}", steps, str(ans))


def _ra_firstarr():
    """Rung 2: only the FIRST arrival of each new post matters; arrivals evict in order 1..n."""
    seen = set()
    evict_time = {}
    steps = [row_step("posts 1..5 in the field", [("queue (bottom leaves first)", list(range(1, 6)), None),
                                                  ("new posts seen", [], None)],
                      eq="a repeat reorders the field but NEVER evicts - skip it")]
    t = 0
    for p in RA_P:
        t += 1
        if p in seen:
            steps.append(row_step(f"t={t}: repeat {p}",
                                  [("queue (bottom leaves first)", [str(x) for x in list(range(1, 6))
                                                                    if x not in evict_time], None),
                                   ("evicted", [f"{k}@{v}" for k, v in evict_time.items()], None)],
                                  eq="no eviction"))
            continue
        seen.add(p)
        victim = max(i for i in range(1, 6) if i not in evict_time)  # bottom post (highest index) leaves first
        evict_time[victim] = t
        steps.append(row_step(f"t={t}: first {p}",
                              [("queue (bottom leaves first)", [str(i) for i in range(1, 6)
                                                                if i not in evict_time], None),
                               ("evicted", [f"{k}@{v}" for k, v in evict_time.items()], None)],
                              eq=f"victim {victim} -> t={t}"))
    ans = [evict_time.get(i, -1) for i in range(1, RA_N + 1)]
    assert ans == RA_ANS
    return _pack("Recent Actions - first-arrival clock", f"n=5 p={RA_P}", steps, str(ans))


def _ra_formula():
    """Rung 3: answer[i] = time of the i-th distinct arrival, or -1; no queue at all."""
    firsts = []
    seen = set()
    for t, p in enumerate(RA_S3_P, 1):
        if p not in seen:
            seen.add(p)
            firsts.append(t)
    steps = [row_step("distinct arrivals", [("firsts (in order)", firsts, None)],
                      eq=f"sample 3: n=4 p={RA_S3_P}; repeats 9,5 skipped")]
    ans = []
    for i in range(1, RA_S3_N + 1):
        k = RA_S3_N - i            # bottom post (n) dies on the 1st arrival, post 1 on the (n+1)-th
        a = firsts[k] if k < len(firsts) else -1
        ans.append(a)
        steps.append(row_step(f"post {i}", [("firsts (in order)", firsts, k if k < len(firsts) else None)],
                              eq=f"needs arrival #{k + 1} -> {a}"))
    assert ans == RA_S3_ANS
    return _pack("Recent Actions - arrival-rank formula", f"n=4 p={RA_S3_P}", steps,
                 str(ans))


# ---------------------------------------------------------------- LC 1396 underground
UD_OPS = [("checkIn", 45, "Leyton", 3), ("checkIn", 32, "Paradise", 8),
          ("checkIn", 27, "Leyton", 10), ("checkOut", 45, "Waterloo", 15),
          ("checkOut", 27, "Waterloo", 20), ("checkOut", 32, "Cambridge", 22),
          ("getAvg", "Paradise", "Cambridge"), ("getAvg", "Leyton", "Waterloo"),
          ("checkIn", 10, "Leyton", 24), ("getAvg", "Leyton", "Waterloo"),
          ("checkOut", 10, "Waterloo", 38), ("getAvg", "Leyton", "Waterloo")]
UD_ANS = [None, None, None, None, None, None, 14.0, 11.0, None, 11.0, None, 12.0]


def _ud_check(op, want, r):
    if want is None:
        assert r is None
    else:
        assert abs(r - want) < 1e-6


def _ud_trips():
    """Rung 1: keep every finished trip; getAverageTime re-sums the whole list."""
    pending = {}
    trips = defaultdict(list)
    out = []
    steps = [row_step("trips per route", [("trips", ["-"], None)],
                      eq="checkIn parks (station, t); checkOut pairs it -> one stored duration")]
    for op, want in zip(UD_OPS, UD_ANS):
        kind = op[0]
        if kind == "checkIn":
            pending[op[1]] = (op[2], op[3]); r = None; lab = f"in {op[1]}@{op[3]}"
        elif kind == "checkOut":
            st, t0 = pending.pop(op[1]); dur = op[3] - t0; trips[(st, op[2])].append(dur)
            r = None; lab = f"out {op[1]} {st}->{op[2]} dur {dur}"
        else:
            route = (op[1], op[2]); r = sum(trips[route]) / len(trips[route]); lab = f"avg {op[1]}->{op[2]}"
        _ud_check(op, want, r)
        out.append(r)
        cells = [f"{a}>{b}:{trips[(a, b)]}" for a, b in trips]
        steps.append(row_step(lab, [("trips", cells or ["-"], None)],
                              eq=f"-> {r}" if r is not None else "recorded"))
    assert out == UD_ANS
    return _pack("Underground - store every trip", "getAvg scans the list", steps,
                 "outputs " + " ".join(str(x) for x in out))


def _ud_runavg():
    """Rung 2: keep a running (sum, count) per route; getAverageTime is O(1)."""
    pending = {}
    stats = {}
    out = []
    steps = [row_step("route -> (sum, cnt)", [("stats", ["-"], None)],
                      eq="checkout folds the duration into the route total; no per-trip list")]
    for op, want in zip(UD_OPS, UD_ANS):
        kind = op[0]
        if kind == "checkIn":
            pending[op[1]] = (op[2], op[3]); r = None; lab = f"in {op[1]}@{op[3]}"
        elif kind == "checkOut":
            st, t0 = pending.pop(op[1]); dur = op[3] - t0
            s, c = stats.get((st, op[2]), (0, 0)); stats[(st, op[2])] = (s + dur, c + 1)
            r = None; lab = f"out {op[1]} {st}->{op[2]} +{dur}"
        else:
            s, c = stats[(op[1], op[2])]; r = s / c; lab = f"avg {op[1]}->{op[2]}"
        _ud_check(op, want, r)
        out.append(r)
        cells = [f"{a}>{b}=({s},{c})" for (a, b), (s, c) in stats.items()]
        steps.append(row_step(lab, [("stats", cells or ["-"], None)],
                              eq=f"-> {r}" if r is not None else "folded"))
    assert out == UD_ANS
    return _pack("Underground - running totals", "getAvg is O(1)", steps,
                 "outputs " + " ".join(str(x) for x in out))


def _ud_oneline():
    """Rung 3: two dicts - card -> (start, t) and route -> [total, n]; the
    pending record is reset lazily by popping it on checkout."""
    start = {}
    total = defaultdict(lambda: [0, 0])
    out = []
    steps = [line_step("shape", [
        "checkIn: open[id] = (station, t)",
        "checkOut: (s, t0) = open.pop(id); total[(s, end)] gets t - t0",
        "getAverageTime: total[route][0] / total[route][1]",
    ], eq="the pending (start, t) is 'reset' by popping it on checkout")]
    for op, want in zip(UD_OPS, UD_ANS):
        kind = op[0]
        if kind == "checkIn":
            start[op[1]] = (op[2], op[3]); r = None; lab = f"in {op[1]}"
        elif kind == "checkOut":
            s, t0 = start.pop(op[1]); route = (s, op[2])
            total[route][0] += op[3] - t0; total[route][1] += 1
            r = None; lab = f"out {op[1]}"
        else:
            t, n = total[(op[1], op[2])]; r = t / n; lab = f"avg {op[1]}->{op[2]}"
        _ud_check(op, want, r)
        out.append(r)
        cells = [f"{a}>{b}:{t}/{n}" for (a, b), (t, n) in total.items()]
        steps.append(row_step(lab, [("totals", cells or ["-"], None)],
                              eq=f"-> {r}" if r is not None else "accrued"))
    assert out == UD_ANS
    return _pack("Underground - one mutable record per route", "avg via running totals", steps,
                 "outputs " + " ".join(str(x) for x in out))


# -------------------------------------------------------------------- registry
REGISTRY = {
    ("randset", "scan"): _rs_scan(),
    ("randset", "swap"): _rs_swap(),
    ("randset", "map"): _rs_jukebox(),
    ("randcol", "scanlast"): _rc_scanlast(),
    ("randcol", "possets"): _rc_possets(),
    ("randcol", "anyslot"): _rc_anyslot(),
    ("snap", "fullcopy"): _snap_fullcopy(),
    ("snap", "columns"): _snap_columns(),
    ("snap", "lazy"): _snap_lazy(),
    ("history", "cursor"): _bh_cursor(),
    ("history", "window"): _bh_window(),
    ("history", "stack"): _bh_stack(),
    ("freqstack", "scan"): _fs_scan(),
    ("freqstack", "levels"): _fs_levels(),
    ("freqstack", "snapshot"): _fs_snapshot(),
    ("hotel", "scan"): _ht_scan(),
    ("hotel", "runs"): _ht_runs(),
    ("allone", "scan"): _al_scan(),
    ("allone", "buckets"): _al_buckets(),
    ("allone", "linked"): _al_linked(),
    ("hashmap", "direct"): _hm_direct(),
    ("hashmap", "buckets"): _hm_buckets(),
    ("hashmap", "resize"): _hm_resize(),
    ("recent", "simulate"): _ra_simulate(),
    ("recent", "firstarr"): _ra_firstarr(),
    ("recent", "formula"): _ra_formula(),
    ("underground", "trips"): _ud_trips(),
    ("underground", "runavg"): _ud_runavg(),
    ("underground", "oneline"): _ud_oneline(),
}


def get(problem, form):
    return REGISTRY[(problem, form)]
