"""Step frames for the intervals & sweep-line chapter.

Drawing lives in ``dsa_viz.story``. ``get`` returns data only. Each story
walks the chapter's own sample, at most eight frames, and the numbers come
from running the algorithm - never from a hand-typed table.
An interval frame shows the interval list with the current interval or the
sweep cursor highlighted plus the rolling chips (cur_end, running counter,
overlap count, chosen set), so the reader watches a *decision* at a
boundary - merge / open / close / pick / drop - exactly as the algorithm
makes it.

Every problem carries the full ladder: a ``brute`` frame set (the direct
definition, often a nested scan over interval pairs or a point-by-point
test), an ``intermediate`` frame set (a sort-based view that gets the shape
right but not yet the insight, or a wrong-greedy counterexample), and an
``optimal`` frame set (the rung the pattern names). ``get`` returns
whichever frame set the caller asks for.
"""
from __future__ import annotations

import bisect
import itertools
import math
import sys
from collections import Counter, defaultdict
from heapq import heappop, heappush
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from trace_log import chip_step, row_step, show  # noqa: E402


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    if not 1 <= len(steps) <= 8:
        raise AssertionError(f"{title}: {len(steps)} frames")
    return (title, inp, steps, text)


def _step(title, rows, chips=None, eq=""):
    """One frame: one or two interval rows plus the rolling chips."""
    frame = row_step(title, rows, eq=eq)
    if chips:
        frame["chips"] = chip_step(title, chips, eq="")["chips"]
    return frame


def _ivals(intervals, hi=None, label="intervals"):
    """A row of intervals shown as strings with the current one highlighted.

    Cells render ``[a,b]``; ``hi`` is an inclusive index range to colour
    amber (the currently examined interval, or the last merged block)."""
    cells = [f"[{a},{b}]" for a, b in intervals]
    color = None
    if hi is not None:
        color = {k: "amber" for k in range(hi[0], hi[1] + 1) if 0 <= k < len(cells)}
    return (label, cells, color)


# ------------------------------------------------------------------ LC 56 merge

def _merge_brute():
    a = [[1, 3], [2, 6], [8, 10], [15, 18]]
    steps = [_step("input", [_ivals(a)], [("pass", 1, "indigo")],
                   eq="repeatedly merge any overlapping pair until stable")]
    cur = [list(x) for x in a]
    pass_no = 1
    for _ in range(4):                    # bounded: never more than n-1 merges
        merged = False
        for i in range(len(cur)):
            for j in range(i + 1, len(cur)):
                if cur[i][0] <= cur[j][1] and cur[j][0] <= cur[i][1]:
                    new = [min(cur[i][0], cur[j][0]), max(cur[i][1], cur[j][1])]
                    cur = [cur[k] for k in range(len(cur)) if k not in (i, j)] + [new]
                    cur.sort()
                    steps.append(_step(f"pass {pass_no}: merge ({i},{j})",
                                       [_ivals(cur, (0, len(cur) - 1))],
                                       [("pass", pass_no, "indigo"), ("size", len(cur), "green")],
                                       eq=f"{new} replaces the pair"))
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
        pass_no += 1
    assert cur == [[1, 6], [8, 10], [15, 18]]
    return _pack("Merge Intervals - brute (pairwise passes)",
                 "intervals = [[1,3],[2,6],[8,10],[15,18]]", steps, str(cur))


def _merge_stack():
    a = [[1, 3], [2, 6], [8, 10], [15, 18]]
    s = sorted(a)
    steps = [_step("sort by start", [_ivals(s)], [("stack", "[]", "muted")],
                   eq="keep a stack of merged blocks; each new interval either extends the top or pushes a new one")]
    stack = []
    for i, iv in enumerate(s):
        if stack and iv[0] <= stack[-1][1]:
            stack[-1] = [stack[-1][0], max(stack[-1][1], iv[1])]
            note = f"extend top -> {stack[-1]}"
        else:
            stack.append(list(iv))
            note = f"push new -> {iv}"
        steps.append(_step(f"see {iv}", [_ivals(stack, (len(stack) - 1, len(stack) - 1))],
                           [("blocks", len(stack), "green"), ("top", str(stack[-1]), "amber")],
                           eq=note))
    assert stack == [[1, 6], [8, 10], [15, 18]]
    return _pack("Merge Intervals - sort + stack of blocks",
                 "sorted [[1,3],[2,6],[8,10],[15,18]]", steps, str(stack))


def _merge_optimal():
    a = [[1, 3], [8, 10], [2, 6], [15, 18]]
    s = sorted(a)
    steps = [_step("sort by start", [_ivals(s)], [("cur_end", "-", "muted")],
                   eq="one pass with a running end; no stack, just the last block")]
    out = []
    for iv in s:
        if out and iv[0] <= out[-1][1]:
            out[-1][1] = max(out[-1][1], iv[1])
            eq = f"overlap: extend end to {out[-1][1]}"
        else:
            out.append(list(iv))
            eq = f"disjoint: start new block [{iv[0]},{iv[1]}]"
        steps.append(_step(f"visit {iv}", [_ivals(out, (len(out) - 1, len(out) - 1))],
                           [("blocks", len(out), "green"), ("cur_end", out[-1][1], "amber")], eq=eq))
    assert out == [[1, 6], [8, 10], [15, 18]]
    return _pack("Merge Intervals - sort by start + running end",
                 "intervals = [[1,3],[8,10],[2,6],[15,18]]", steps, str(out))


# ------------------------------------------------------------------ LC 57 insert

def _insert_brute():
    a = [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]
    new = [4, 8]
    joined = sorted([list(x) for x in a] + [list(new)])
    steps = [_step("concat then re-merge", [_ivals(joined)], [("size", len(joined), "indigo")],
                   eq="throw newInterval in and run the full merge again")]
    out = []
    for iv in joined:
        if out and iv[0] <= out[-1][1]:
            out[-1][1] = max(out[-1][1], iv[1])
        else:
            out.append(list(iv))
    steps.append(_step("after merge", [_ivals(out, (0, len(out) - 1))],
                       [("blocks", len(out), "green")], eq=str(out)))
    assert out == [[1, 2], [3, 10], [12, 16]]
    return _pack("Insert Interval - brute (concat + full merge)",
                 "intervals=[[1,2],[3,5],[6,7],[8,10],[12,16]] new=[4,8]", steps, str(out))


def _insert_linear():
    a = [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]
    new = [4, 8]
    steps = [_step("three phases", [_ivals(a)], [("phase", "before", "indigo")],
                   eq="emit strictly-left, absorb overlaps, emit strictly-right")]
    out, i = [], 0
    while i < len(a) and a[i][1] < new[0]:
        out.append(list(a[i]))
        steps.append(_step(f"phase 1: emit {a[i]}", [_ivals(out, (len(out) - 1, len(out) - 1))],
                           [("phase", "before", "indigo")], eq=f"end {a[i][1]} < start {new[0]}"))
        i += 1
    cur = list(new)
    while i < len(a) and a[i][0] <= cur[1]:
        cur = [min(cur[0], a[i][0]), max(cur[1], a[i][1])]
        steps.append(_step(f"phase 2: absorb {a[i]}", [_ivals(out, (0, len(out) - 1))],
                           [("phase", "merge", "amber"), ("cur", str(cur), "green")],
                           eq=f"cur := {cur}"))
        i += 1
    out.append(cur)
    while i < len(a):
        out.append(list(a[i]))
        steps.append(_step(f"phase 3: emit {a[i]}", [_ivals(out, (len(out) - 1, len(out) - 1))],
                           [("phase", "after", "indigo")], eq=""))
        i += 1
    if len(steps) > 8:
        steps = steps[:4] + steps[-3:]
    assert out == [[1, 2], [3, 10], [12, 16]]
    return _pack("Insert Interval - linear three-phase scan",
                 "intervals=[[1,2],[3,5],[6,7],[8,10],[12,16]] new=[4,8]", steps, str(out))


def _insert_bisect():
    a = [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]
    new = [4, 8]
    starts = [x[0] for x in a]
    ends = [x[1] for x in a]
    lo = bisect.bisect_left(ends, new[0])         # first interval with end >= new.start
    hi = bisect.bisect_right(starts, new[1])      # first interval with start > new.end
    merged = [min(new[0], a[lo][0]), max(new[1], a[hi - 1][1])]
    out = a[:lo] + [merged] + a[hi:]
    steps = [
        _step("sorted list", [_ivals(a)], [("new", str(new), "indigo")],
              eq="two binary searches bracket the run to absorb"),
        _step(f"lo = bisect_left(ends, {new[0]})", [_ivals(a, (lo, lo))],
              [("lo", lo, "amber")], eq=f"first interval whose end >= {new[0]}"),
        _step(f"hi = bisect_right(starts, {new[1]})", [_ivals(a, (hi - 1, hi - 1))],
              [("hi", hi, "amber")], eq=f"first start strictly > {new[1]}"),
        _step("splice [lo..hi) with merged", [_ivals([list(x) for x in out], (lo, lo))],
              [("merged", str(merged), "green")], eq=str(merged)),
    ]
    assert out == [[1, 2], [3, 10], [12, 16]]
    return _pack("Insert Interval - two bisects + splice",
                 "intervals=[[1,2],[3,5],[6,7],[8,10],[12,16]] new=[4,8]", steps, str(out))


# ------------------------------------------------------------------ CSES 1619 restaurant

_REST_ARR = [1, 2, 4]
_REST_DEP = [5, 6, 8]


def _rest_pointwise():
    arr, dep = _REST_ARR, _REST_DEP
    iv = list(zip(arr, dep))
    steps = [_step("couples", [_ivals(iv)], [("best", 0, "muted")],
                   eq="test each arrival time as the query point")]
    best = 0
    for t in arr:
        c = sum(1 for a, d in iv if a <= t <= d)
        best = max(best, c)
        steps.append(_step(f"t = {t}", [_ivals(iv)],
                           [("t", t, "indigo"), ("count", c, "amber"), ("best", best, "green")],
                           eq=f"{c} intervals contain {t}"))
    assert best == 3
    return _pack("Restaurant Customers - test each arrival",
                 "arr=[1,2,4] dep=[5,6,8]", steps, str(best))


def _rest_events():
    arr, dep = _REST_ARR, _REST_DEP
    # Sort by (time, -delta): on a tie an arrival (+1) precedes a departure (-1),
    # the closed reading used by the chapter's code (CSES itself has no ties).
    evs = sorted([(a, +1) for a in arr] + [(d, -1) for d in dep], key=lambda e: (e[0], -e[1]))
    steps = [_step("sorted events", [(None, [f"{t}{'+' if k > 0 else '-'}" for t, k in evs], None)],
                   [("cur", 0, "muted"), ("best", 0, "muted")],
                   eq="walk events; +1 arrival, -1 departure; ties: arrival first")]
    cur = best = 0
    for i, (t, k) in enumerate(evs):
        cur += k
        if cur > best:
            best = cur
        cols = {i: "amber"}
        steps.append(_step(f"event {i}: t={t}, d={k:+d}",
                           [(None, [f"{tt}{'+' if kk > 0 else '-'}" for tt, kk in evs], cols)],
                           [("cur", cur, "amber"), ("best", best, "green")],
                           eq=f"running count {cur}"))
    assert best == 3
    return _pack("Restaurant Customers - event sweep",
                 f"events = {evs}", steps, str(best))


def _rest_diff():
    arr, dep = _REST_ARR, _REST_DEP
    coord = sorted(set(arr) | set(dep))
    idx = {v: i for i, v in enumerate(coord)}
    diff = [0] * (len(coord) + 1)
    for a, d in zip(arr, dep):
        diff[idx[a]] += 1
        diff[idx[d] + 1] -= 1
    steps = [_step("diff array on compressed coords",
                   [(None, [f"{c}:{diff[i]:+d}" for i, c in enumerate(coord)] + [f"sent:{diff[len(coord)]:+d}"], None)],
                   [("best", 0, "muted")],
                   eq="+1 at arrival, -1 just past departure")]
    run = best = 0
    occ = []
    for i, c in enumerate(coord):
        run += diff[i]
        best = max(best, run)
        occ.append(run)
        steps.append(_step(f"coord {c}",
                           [(None, [str(o) for o in occ] + [""] * (len(coord) - len(occ)), {i: "amber"})],
                           [("at", c, "indigo"), ("cur", run, "amber"), ("best", best, "green")],
                           eq=f"{run} customers at {c}"))
    assert best == 3
    return _pack("Restaurant Customers - difference array on coords",
                 "arr=[1,2,4] dep=[5,6,8]", steps, str(best))


# ------------------------------------------------------------------ LC 1094 car pooling

def _pool_per_mile():
    trips = [[2, 1, 5], [3, 3, 7]]
    cap = 4
    steps = [_step("trips", [(None, [f"{p}@[{f},{t})" for p, f, t in trips], None)],
               [("cap", cap, "muted")],
               eq="[passengers, from, to]; find if any mile has load > cap")]
    ok = True
    peak = 0
    for mile in (1, 2, 3, 4, 5, 6):
        load = sum(p for p, f, t in trips if f <= mile < t)
        peak = max(peak, load)
        if load > cap:
            ok = False
        steps.append(_step(f"mile {mile}", [],
                           [("mile", mile, "indigo"), ("load", load, "amber" if load <= cap else "red"),
                            ("peak", peak, "green")],
                           eq=f"{load} passengers onboard"))
    final = "true" if ok else "false"
    assert peak == 5 and not ok
    return _pack("Car Pooling - per-mile scan",
                 "trips=[[2,1,5],[3,3,7]] capacity=4", steps, final)


def _pool_events():
    trips = [[2, 1, 5], [3, 3, 7]]
    cap = 4
    # sort key (loc, is_pickup): drop events (0) come BEFORE pickup (1) at same loc.
    evs = sorted([(f, 1, p) for p, f, t in trips] + [(t, 0, p) for p, f, t in trips])
    steps = [_step("events (drop before pickup at same loc)",
                   [(None, [f"{loc}:{('+' if kind else '-')}{p}" for loc, kind, p in evs], None)],
                   [("cap", cap, "muted")],
                   eq="sweep positions; pickup adds p, drop subtracts p")]
    load = 0
    peak = 0
    ok = True
    for i, (loc, kind, p) in enumerate(evs):
        load += p if kind else -p
        peak = max(peak, load)
        if load > cap:
            ok = False
        cols = {i: "amber"}
        steps.append(_step(f"event {i}: loc {loc}, {('pickup' if kind else 'drop')}",
                           [(None, [f"{a}:{('+' if b else '-')}{c}" for a, b, c in evs], cols)],
                           [("load", load, "amber" if load <= cap else "red"),
                            ("peak", peak, "green")],
                           eq=f"onboard {load}"))
    final = "true" if ok else "false"
    assert peak == 5 and not ok
    return _pack("Car Pooling - event sweep",
                 "trips=[[2,1,5],[3,3,7]] capacity=4", steps, final)


def _pool_diff():
    trips = [[2, 1, 5], [3, 3, 7]]
    cap = 4
    diff = [0] * 11
    for p, f, t in trips:
        diff[f] += p
        diff[t] -= p
    steps = [_step("diff on mile markers 0..10",
                   [(None, [str(x) for x in diff], None)],
                   [("cap", cap, "muted")],
                   eq="+p at start, -p at end (half-open [from,to))")]
    run = peak = 0
    ok = True
    for mile in (1, 3, 5, 7):
        for m in range(max(1, mile - 1), mile + 1):
            run += diff[m]
        peak = max(peak, run)
        if run > cap:
            ok = False
        steps.append(_step(f"prefix at mile {mile}",
                           [(None, [str(diff[m]) for m in range(11)], {mile: "amber"})],
                           [("mile", mile, "indigo"), ("load", run, "amber" if run <= cap else "red"),
                            ("peak", peak, "green")],
                           eq=f"cumulative {run}"))
    final = "true" if ok else "false"
    assert peak == 5 and not ok
    return _pack("Car Pooling - difference array",
                 "trips=[[2,1,5],[3,3,7]] capacity=4", steps, final)


# ------------------------------------------------------------------ CSES 1629 movie festival

# The three boards share the SAME input. Sorting by start picks (1,100) and
# stops at one movie - a beautiful wrong-greedy counterexample - whereas the
# exhaustive brute and the earliest-end greedy both reach three.
_FEST = [(1, 100), (2, 3), (4, 5), (6, 7)]


def _fest_subsets():
    iv = _FEST
    steps = [_step("movies", [_ivals(iv)], [("n", len(iv), "indigo")],
                   eq="try every subset (2^4=16); keep the largest with no pairwise overlap")]
    best = 0
    seen = []
    for r in range(1, len(iv) + 1):
        for combo in itertools.combinations(range(len(iv)), r):
            times = [iv[i] for i in combo]
            ok = all(a[1] <= b[0] or b[1] <= a[0]
                     for a, b in itertools.combinations(times, 2))
            if ok and len(times) > best:
                best = len(times)
                seen.append((str(times), len(times), best))
    for label, size, val in seen[:5]:
        steps.append(_step(f"feasible {label}", [_ivals(iv)],
                           [("size", size, "green"), ("best", val, "amber")],
                           eq=f"new best {val}"))
    if len(steps) > 8:
        steps = steps[:4] + steps[-3:]
    assert best == 3
    return _pack("Movie Festival - subset enumeration",
                 "movies = [(1,100),(2,3),(4,5),(6,7)]", steps, str(best))


def _fest_by_start():
    iv = _FEST
    s = sorted(iv)
    steps = [_step("sort by start", [_ivals(s)], [("last_end", 0, "muted"), ("chosen", 0, "muted")],
                   eq="WRONG greedy: pick every movie whose start is after the last chosen end")]
    last_end = -1
    chosen = []
    for i, (a, b) in enumerate(s):
        if a > last_end:
            chosen.append((a, b))
            last_end = b
            note = f"take ({a},{b}); last_end={b}"
        else:
            note = f"skip ({a},{b}); start {a} <= {last_end}"
        hi_idx = len(chosen) - 1 if chosen and chosen[-1] == (a, b) else None
        steps.append(_step(f"see ({a},{b})", [_ivals(s, (i, i))],
                           [("chosen", len(chosen), "green"), ("last_end", last_end, "amber")],
                           eq=note))
    assert len(chosen) == 1                     # (1,100) blocks everything else
    return _pack("Movie Festival - sort by start (a wrong greedy)",
                 "sorted by start [(1,100),(2,3),(4,5),(6,7)]", steps, str(len(chosen)))


def _fest_by_end():
    iv = _FEST
    s = sorted(iv, key=lambda x: x[1])
    steps = [_step("sort by end", [_ivals(s)], [("last_end", 0, "muted"), ("chosen", 0, "muted")],
                   eq="greedy: take the movie that FREES the room earliest")]
    last_end = -1
    chosen = []
    for i, (a, b) in enumerate(s):
        if a > last_end:
            chosen.append((a, b))
            last_end = b
            note = f"take ({a},{b}); last_end={b}"
        else:
            note = f"skip ({a},{b}); start {a} <= {last_end}"
        steps.append(_step(f"see ({a},{b})", [_ivals(s, (i, i))],
                           [("chosen", len(chosen), "green"), ("last_end", last_end, "amber")],
                           eq=note))
    assert len(chosen) == 3                     # (2,3),(4,5),(6,7); (1,100) skipped
    return _pack("Movie Festival - sort by end (correct greedy)",
                 "sorted by end [(2,3),(4,5),(6,7),(1,100)]", steps, str(len(chosen)))


# ------------------------------------------------------------------ CSES 1632 movie festival II

_FEST2 = [(1, 4), (2, 5), (3, 6), (5, 8), (7, 9)]
_FEST2_K = 2


def _fest2_repeat():
    iv, k = _FEST2, _FEST2_K
    steps = [_step("movies, k rooms", [_ivals(iv)], [("k", k, "indigo")],
                   eq="run single-room greedy k times over the leftovers")]
    remaining = list(iv)
    total = 0
    for room in range(k):
        s = sorted(remaining, key=lambda x: x[1])
        last_end = 0
        used = []
        for a, b in s:
            if a >= last_end:
                used.append((a, b))
                last_end = b
        for u in used:
            remaining.remove(u)
        total += len(used)
        steps.append(_step(f"room {room + 1}", [_ivals(used, (0, len(used) - 1))],
                           [("total", total, "green"), ("left", len(remaining), "amber")],
                           eq=f"{len(used)} movies assigned"))
    assert total == 4
    return _pack("Movie Festival II - run single-room greedy k times",
                 f"movies={iv} k={k}", steps, str(total))


def _fest2_heap():
    iv, k = _FEST2, _FEST2_K
    s = sorted(iv)
    steps = [_step("sort by start", [_ivals(s)], [("k", k, "indigo"), ("done", 0, "muted")],
                   eq="min-heap of currently-busy room end times; evict any room free before this start")]
    heap = []
    done = 0
    for i, (a, b) in enumerate(s):
        while heap and heap[0] <= a:
            heappop(heap)
        if len(heap) < k:
            heappush(heap, b)
            done += 1
            note = f"take ({a},{b}); rooms busy {len(heap)}"
        else:
            note = f"skip ({a},{b}); no room free"
        steps.append(_step(f"see ({a},{b})",
                           [(None, [str(x) for x in sorted(heap)], None),
                            _ivals(s, (i, i))],
                           [("heap", str(sorted(heap)), "amber"), ("done", done, "green")], eq=note))
    assert done == 4
    return _pack("Movie Festival II - sort by start + room-heap (subtle-wrong sweep)",
                 f"movies={iv} k={k}", steps, str(done))


def _fest2_multiset():
    iv, k = _FEST2, _FEST2_K
    s = sorted(iv, key=lambda x: x[1])
    steps = [_step("sort by end", [_ivals(s)], [("k", k, "indigo"), ("done", 0, "muted")],
                   eq="each movie tries the room with the LATEST free-time still <= its start")]
    free = [0] * k
    done = 0
    for i, (a, b) in enumerate(s):
        best_idx = -1
        for j, t in enumerate(free):
            if t <= a and (best_idx == -1 or free[best_idx] < t):
                best_idx = j
        if best_idx >= 0:
            free[best_idx] = b
            done += 1
            note = f"assign ({a},{b}) to room {best_idx}"
        else:
            note = f"skip ({a},{b}); no room free"
        steps.append(_step(f"see ({a},{b})",
                           [_ivals(s, (i, i)), (None, [f"r{j}:{t}" for j, t in enumerate(free)], None)],
                           [("rooms", str(free), "amber"), ("done", done, "green")], eq=note))
    assert done == 4
    return _pack("Movie Festival II - sort by end + multiset",
                 f"movies={iv} k={k}", steps, str(done))


# ------------------------------------------------------------------ LC 452 arrows

_ARROWS = [(10, 16), (2, 8), (1, 6), (7, 12)]


def _arrows_brute():
    iv = _ARROWS
    ends = sorted({b for _, b in iv})
    steps = [_step("candidates = endpoints", [_ivals(iv)],
                   [("endpoints", str(ends), "indigo")],
                   eq="an optimal arrow can always sit at an interval ENDPOINT")]
    for x in ends:
        c = sum(1 for a, b in iv if a <= x <= b)
        steps.append(_step(f"arrow at {x}", [_ivals(iv)],
                           [("arrow", x, "indigo"), ("cover", c, "amber")],
                           eq=f"{c} balloons contain {x}"))
    # min-cover over subsets of endpoints (2^4=16, tiny).
    best = None
    for r in range(1, len(ends) + 1):
        for combo in itertools.combinations(ends, r):
            if all(any(a <= x <= b for x in combo) for a, b in iv):
                best = list(combo)
                break
        if best:
            break
    steps.append(_step(f"min subset: {best}", [_ivals(iv)],
                       [("arrows", len(best), "green")],
                       eq=f"{len(best)} arrows burst every balloon"))
    assert len(best) == 2
    return _pack("Minimum Arrows - test endpoints, then min subset cover",
                 f"points = {iv}", steps, str(len(best)))


def _arrows_chain():
    iv = _ARROWS
    s = sorted(iv)
    steps = [_step("sort by start", [_ivals(s)], [("arrows", 1, "indigo"),
                                                    ("window", str(s[0]), "amber")],
                   eq="carry the current intersection; when it empties, spend a new arrow")]
    lo, hi = s[0]
    arrows = 1
    for i, (a, b) in enumerate(s[1:], start=1):
        if a > hi:
            arrows += 1
            lo, hi = a, b
            note = f"disjoint -> new arrow, window [{lo},{hi}]"
        else:
            lo = max(lo, a)
            hi = min(hi, b)
            note = f"intersect -> tighten window [{lo},{hi}]"
        steps.append(_step(f"see ({a},{b})", [_ivals(s, (i, i))],
                           [("arrows", arrows, "green"), ("window", f"[{lo},{hi}]", "amber")], eq=note))
    assert arrows == 2
    return _pack("Minimum Arrows - sort by start + chain intersection",
                 "sorted [(1,6),(2,8),(7,12),(10,16)]", steps, str(arrows))


def _arrows_by_end():
    iv = _ARROWS
    s = sorted(iv, key=lambda x: x[1])
    steps = [_step("sort by end", [_ivals(s)], [("arrows", 1, "indigo"), ("shot", s[0][1], "amber")],
                   eq="greedy: shoot at the earliest end; skip every balloon it bursts")]
    shot = s[0][1]
    arrows = 1
    for i, (a, b) in enumerate(s[1:], start=1):
        if a > shot:
            arrows += 1
            shot = b
            note = f"({a},{b}) missed -> new arrow at {b}"
        else:
            note = f"({a},{b}) already burst (start {a} <= {shot})"
        steps.append(_step(f"see ({a},{b})", [_ivals(s, (i, i))],
                           [("arrows", arrows, "green"), ("shot", shot, "amber")], eq=note))
    assert arrows == 2
    return _pack("Minimum Arrows - sort by end + greedy shot",
                 "sorted by end [(1,6),(2,8),(7,12),(10,16)]", steps, str(arrows))


# ------------------------------------------------------------------ LC 1288 remove covered

_COVERED = [(1, 4), (2, 3), (3, 6), (4, 5)]


def _covered_pairwise():
    iv = _COVERED
    steps = [_step("ranges", [_ivals(iv)], [("keep", 0, "muted")],
                   eq="for each i, test if some other j contains it: a_j <= a_i and b_i <= b_j")]
    keep = []
    for i, (a, b) in enumerate(iv):
        cov = any(a2 <= a and b <= b2 for j, (a2, b2) in enumerate(iv) if j != i)
        if not cov:
            keep.append((a, b))
        steps.append(_step(f"check ({a},{b})", [_ivals(iv, (i, i))],
                           [("covered", "yes" if cov else "no", "amber" if cov else "green"),
                            ("keep", len(keep), "green")],
                           eq="contained by some other" if cov else "kept"))
    assert len(keep) == 2                       # (1,4) and (3,6) survive
    return _pack("Remove Covered - pairwise containment",
                 f"ranges={iv}", steps, str(len(keep)))


def _covered_sweep():
    iv = _COVERED
    s = sorted(iv, key=lambda x: (x[0], -x[1]))
    steps = [_step("sort start asc, end desc", [_ivals(s)],
                   [("max_end", 0, "muted"), ("keep", 0, "muted")],
                   eq="an interval is covered iff its end <= the running max_end")]
    max_end = 0
    keep = 0
    for i, (a, b) in enumerate(s):
        if b > max_end:
            keep += 1
            max_end = b
            note = f"({a},{b}) extends to {b} -> keep"
        else:
            note = f"({a},{b}) inside [..,{max_end}] -> covered"
        steps.append(_step(f"see ({a},{b})", [_ivals(s, (i, i))],
                           [("keep", keep, "green"), ("max_end", max_end, "amber")], eq=note))
    assert keep == 2
    return _pack("Remove Covered - sort + running max end",
                 "sorted [(1,4),(2,3),(3,6),(4,5)]", steps, str(keep))


def _covered_by_end():
    iv = _COVERED
    s = sorted(iv, key=lambda x: (x[1], -x[0]))
    steps = [_step("sort end asc, start desc", [_ivals(s)],
                   [("min_start", math.inf, "muted"), ("keep", 0, "muted")],
                   eq="dual: scan right-to-left with a running min_start; earlier is covered iff its start >= min_start")]
    # Reverse scan: from the interval with the largest end toward the smallest.
    rev = list(reversed(s))
    min_start = math.inf
    keep = 0
    for i, (a, b) in enumerate(rev):
        if a < min_start:
            keep += 1
            min_start = a
            note = f"({a},{b}) extends left -> keep"
        else:
            note = f"({a},{b}) covered by later with start <= {a}"
        steps.append(_step(f"see ({a},{b})", [_ivals(rev, (i, i))],
                           [("keep", keep, "green"), ("min_start", min_start, "amber")], eq=note))
    assert keep == 2
    return _pack("Remove Covered - sort by end + right-to-left min_start",
                 "reverse-scanning [(4,5),(3,6),(2,3),(1,4)]", steps, str(keep))


# ------------------------------------------------------------------ LC 986 interval intersections

_A = [(0, 2), (5, 10), (13, 23), (24, 33)]
_B = [(1, 5), (8, 12), (15, 24), (25, 26)]
_INT_EXPECTED = [(1, 2), (5, 5), (8, 10), (15, 23), (24, 24), (25, 26)]


def _int_pairs():
    A, B = _A, _B
    steps = [_step("two lists", [_ivals(A, label="A"), _ivals(B, label="B")],
                   [("hits", 0, "muted")], eq="intersect every pair (|A| x |B| checks)")]
    out = []
    for i, (a1, a2) in enumerate(A):
        for j, (b1, b2) in enumerate(B):
            lo, hi = max(a1, b1), min(a2, b2)
            if lo <= hi:
                out.append((lo, hi))
                steps.append(_step(f"A[{i}] x B[{j}] -> ({lo},{hi})",
                                   [_ivals(A, (i, i), "A"), _ivals(B, (j, j), "B")],
                                   [("hits", len(out), "green")], eq=f"overlap [{lo},{hi}]"))
    if len(steps) > 8:
        steps = steps[:4] + steps[-3:]
    assert out == _INT_EXPECTED
    return _pack("Interval Intersections - every pair",
                 "A=[[0,2],[5,10],[13,23],[24,33]] B=[[1,5],[8,12],[15,24],[25,26]]",
                 steps, str(out))


def _int_two_ptr():
    A, B = _A, _B
    steps = [_step("two sorted lists", [_ivals(A, label="A"), _ivals(B, label="B")],
                   [("i", 0, "indigo"), ("j", 0, "indigo")],
                   eq="advance whichever pointer has the smaller end")]
    i = j = 0
    out = []
    while i < len(A) and j < len(B):
        lo, hi = max(A[i][0], B[j][0]), min(A[i][1], B[j][1])
        if lo <= hi:
            out.append((lo, hi))
            eq = f"overlap [{lo},{hi}]"
        else:
            eq = "no overlap"
        if A[i][1] < B[j][1]:
            i += 1
            moved = "advance i"
        else:
            j += 1
            moved = "advance j"
        steps.append(_step(f"cursor A[{min(i, len(A)-1)}] x B[{min(j, len(B)-1)}]",
                           [_ivals(A, (min(i, len(A) - 1), min(i, len(A) - 1)), "A"),
                            _ivals(B, (min(j, len(B) - 1), min(j, len(B) - 1)), "B")],
                           [("i", i, "amber"), ("j", j, "amber"), ("hits", len(out), "green")],
                           eq=f"{eq}; {moved}"))
        if len(steps) >= 8:
            break
    assert out == _INT_EXPECTED
    return _pack("Interval Intersections - two-pointer sweep",
                 "A=[[0,2],[5,10],[13,23],[24,33]] B=[[1,5],[8,12],[15,24],[25,26]]",
                 steps, str(out))


def _int_events():
    A, B = _A, _B
    ev = []
    for a, b in A:
        ev += [(a, "open-A"), (b + 1, "close-A")]
    for a, b in B:
        ev += [(a, "open-B"), (b + 1, "close-B")]
    ev.sort()
    short = [f"{t}{k[-1]}{'+' if k.startswith('open') else '-'}" for t, k in ev]

    def window(i):
        lo = max(0, min(i - 2, len(ev) - 5))
        return (None, short[lo:lo + 5], {i - lo: "amber"})

    steps = [_step("events", [(None, short[:5] + ["..."], None)],
                   [("inA", 0, "muted"), ("inB", 0, "muted")],
                   eq="close events sit at end+1; emit [prev, t-1] while both lists are inside")]
    inA = inB = 0
    prev = None
    out = []
    for i, (t, kind) in enumerate(ev):
        emitted = False
        if prev is not None and inA and inB and t > prev:
            out.append((prev, t - 1))
            emitted = True
        if kind == "open-A":
            inA += 1
        elif kind == "close-A":
            inA -= 1
        elif kind == "open-B":
            inB += 1
        else:
            inB -= 1
        if emitted:                                   # one frame per emitted segment
            steps.append(_step(f"event {kind}@{t}", [window(i)],
                               [("inA", inA, "indigo"), ("inB", inB, "violet"),
                                ("emit", f"[{out[-1][0]},{out[-1][1]}]", "green")],
                               eq=f"both were inside on [{out[-1][0]},{out[-1][1]}]"))
        prev = t
    assert out == _INT_EXPECTED
    return _pack("Interval Intersections - event sweep",
                 "A=[[0,2],[5,10],[13,23],[24,33]] B=[[1,5],[8,12],[15,24],[25,26]]", steps, str(out))


# ------------------------------------------------------------------ registry

REGISTRY = {
    ("merge", "brute"): _merge_brute(),
    ("merge", "stack"): _merge_stack(),
    ("merge", "optimal"): _merge_optimal(),
    ("insert", "brute"): _insert_brute(),
    ("insert", "linear"): _insert_linear(),
    ("insert", "bisect"): _insert_bisect(),
    ("restaurant", "brute"): _rest_pointwise(),
    ("restaurant", "events"): _rest_events(),
    ("restaurant", "diff"): _rest_diff(),
    ("pooling", "brute"): _pool_per_mile(),
    ("pooling", "events"): _pool_events(),
    ("pooling", "diff"): _pool_diff(),
    ("festival", "brute"): _fest_subsets(),
    ("festival", "wrong"): _fest_by_start(),
    ("festival", "optimal"): _fest_by_end(),
    ("festival2", "brute"): _fest2_repeat(),
    ("festival2", "heap"): _fest2_heap(),
    ("festival2", "multiset"): _fest2_multiset(),
    ("arrows", "brute"): _arrows_brute(),
    ("arrows", "chain"): _arrows_chain(),
    ("arrows", "optimal"): _arrows_by_end(),
    ("covered", "brute"): _covered_pairwise(),
    ("covered", "sweep"): _covered_sweep(),
    ("covered", "dual"): _covered_by_end(),
    ("intersections", "brute"): _int_pairs(),
    ("intersections", "twoptr"): _int_two_ptr(),
    ("intersections", "events"): _int_events(),
}


def get(problem, form):
    return REGISTRY[(problem, form)]
