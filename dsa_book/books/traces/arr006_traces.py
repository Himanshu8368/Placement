"""Step frames for the prefix-sum chapter.

Drawing lives in ``dsa_viz.story``. ``get`` returns data only.
Each story walks the chapter's own sample, at most eight frames, and the
numbers come from running the algorithm — never from a hand-typed table.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import chip_step, grid_step, row_step, show


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    if not 1 <= len(steps) <= 8:
        raise AssertionError(f"{title}: {len(steps)} frames")
    return (title, inp, steps, text)


def _grown(full, filled):
    """The first ``filled`` cells of ``full``, the rest as unread."""
    return list(full[:filled]) + ["?"] * (len(full) - filled)


def _prefix(a):
    p = [0]
    for x in a:
        p.append(p[-1] + x)
    return p


def _seen_map(seen):
    return "{" + ", ".join(f"{k}: {v}" for k, v in sorted(seen.items())) + "}"


def _range_table():
    a = [1, 2, 3, 4, 5]
    queries = [(1, 2), (2, 4), (3, 5)]
    p = _prefix(a)
    steps = [row_step("the table starts empty but for zero",
                      [("a", a, None), ("P", _grown(p, 1), 0)],
                      "P[0] = 0")]
    for i in (0, 2, 4):
        steps.append(row_step(f"P[{i + 1}] = P[{i}] + a[{i}]",
                              [("a", a, i), ("P", _grown(p, i + 2), i + 1)],
                              f"{p[i]} + {a[i]} = {p[i + 1]}"))
    for x, y in queries:
        steps.append(row_step(f"query {x}..{y} costs two reads",
                              [("P", p, y)],
                              f"P[{y}] - P[{x - 1}] = {p[y]} - {p[x - 1]} = {p[y] - p[x - 1]}"))
    ans = [p[y] - p[x - 1] for x, y in queries]
    return _pack("Static Range Sum Queries - one table",
                 "a = [1, 2, 3, 4, 5], queries = [(1,2), (2,4), (3,5)]", steps, ans)


def _pivot_table():
    a = [1, 7, 3, 6, 5, 6]
    p, n = _prefix(a), len(a)
    steps = [row_step("prefix table of the whole array", [("P", p, None)],
                      f"P[n] = {p[n]}")]
    for i in range(4):
        right = p[n] - p[i + 1]
        steps.append(row_step(f"cut at i = {i}: left vs right",
                              [("P", p, i)],
                              f"P[{i}] = {p[i]},  P[n] - P[{i + 1}] = {right}"
                              + ("   equal" if p[i] == right else "")))
    best = next((i for i in range(n) if p[i] == p[n] - p[i + 1]), -1)
    return _pack("Find Pivot Index - table form", "a = [1, 7, 3, 6, 5, 6]", steps, best)


def _pivot_scan():
    a = [1, 7, 3, 6, 5, 6]
    total, left = sum(a), 0
    steps = [chip_step("one total, one running left",
                       [("total", total, "muted"), ("left", 0, "indigo")],
                       "left == total - left - a[i]")]
    for i in range(4):
        x = a[i]
        right = total - left - x
        ok = left == right
        steps.append(chip_step(f"i = {i}, a[i] = {x}",
                               [("left", left, "green" if ok else "indigo"),
                                ("right", right, "green" if ok else "amber"),
                                ("i", i, "muted")],
                               f"{left} == {right}: pivot" if ok
                               else f"{left} != {right}: add {x} into left"))
        left += x
    return _pack("Find Pivot Index - one pass", "a = [1, 7, 3, 6, 5, 6]", steps, 3)


def _sumk_map():
    a, k = [1, 2, 3, -2, 4, 1], 3
    seen = {0: 1}
    p = ans = 0
    steps = [chip_step("seed the empty prefix",
                       [("p", 0, "indigo"), ("ans", 0, "muted")],
                       f"seen = {_seen_map(seen)}")]
    for x in a:
        p += x
        before = seen.get(p - k, 0)
        ans += before
        seen[p] = seen.get(p, 0) + 1
        steps.append(chip_step(f"x = {x}, look for p - k",
                               [("p", p, "indigo"),
                                (f"seen[{p - k}]", before, "amber" if before else "muted"),
                                ("ans", ans, "green")],
                               f"seen = {_seen_map(seen)}"))
    return _pack("Subarray Sum Equals K - hash map", "a = [1,2,3,-2,4,1], k = 3", steps, ans)


def _divk_buckets():
    a, k = [4, 5, 0, -2, -3, 1], 5
    cnt = [0] * k
    cnt[0] = 1
    r = ans = 0
    steps = [row_step("k buckets, empty prefix in bucket 0",
                      [("a", a, None), ("cnt", cnt[:], 0)],
                      "cnt[0] = 1")]
    for i, x in enumerate(a):
        r = (r + x) % k
        before = cnt[r]
        ans += before
        cnt[r] += 1
        steps.append(row_step(f"i = {i}: remainder {r}",
                              [("a", a, i), ("cnt", cnt[:], r)],
                              f"ans += {before} -> {ans},  cnt[{r}] = {cnt[r]}"))
    return _pack("Subarray Sums Divisible by K - buckets",
                 "a = [4, 5, 0, -2, -3, 1], k = 5", steps, ans)


def _good_prefix():
    s = "131"
    seen = {0: 1}
    p = ans = 0
    steps = [chip_step("each digit contributes d - 1",
                       [("p", 0, "indigo"), ("ans", 0, "muted")],
                       f"seen = {_seen_map(seen)};  equal prefixes bound a good substring")]
    for ch in s:
        d = int(ch) - 1
        p += d
        before = seen.get(p, 0)
        ans += before
        seen[p] = before + 1
        steps.append(chip_step(f"'{ch}' adds {d:+d}",
                               [("p", p, "indigo"),
                                ("seen[p]", before, "amber" if before else "muted"),
                                ("ans", ans, "green")],
                               f"seen = {_seen_map(seen)}"))
    return _pack("Good Subarrays - sum equals length", "s = \"131\"", steps, ans)


def _flight_diff():
    n = 5
    bookings = [[1, 2, 10], [2, 3, 20], [2, 5, 25]]
    d = [0] * (n + 1)
    steps = [row_step("two point writes, never the range",
                      [("d", d[:], None)], "n = 5, so d has a spare sixth cell")]
    for first, last, seats in bookings:
        d[first - 1] += seats
        d[last] -= seats
        steps.append(row_step(f"book flights {first}..{last} for {seats}",
                              [("d", d[:], first - 1)],
                              f"d[{first - 1}] += {seats},  d[{last}] -= {seats}"))
    out, run = [], 0
    for i in range(n):
        run += d[i]
        out.append(run)
        if i in (1, 4):
            steps.append(row_step(f"rebuild: out[{i}] = {run}",
                                  [("d", d[:], i), ("out", _grown(out, i + 1), i)],
                                  f"run += d[{i}] -> {run}"))
    return _pack("Corporate Flight Bookings - difference array",
                 "n = 5, bookings = [[1,2,10], [2,3,20], [2,5,25]]", steps, out)


_GREG_A = [1, 2, 3]
_GREG_OPS = [(1, 2, 1), (1, 3, 2), (2, 3, 4)]
_GREG_QUERIES = [(1, 2), (1, 3), (2, 3)]


def _greg_times():
    m = len(_GREG_OPS)
    od = [0] * (m + 1)
    steps = [row_step("the axis is operation numbers",
                      [("odiff", od[:], None)], "m = 3 operations, k = 3 queries")]
    for x, y in _GREG_QUERIES:
        od[x - 1] += 1
        od[y] -= 1
        steps.append(row_step(f"query ({x}, {y}) covers ops {x}..{y}",
                              [("odiff", od[:], x - 1)],
                              f"odiff[{x - 1}] += 1,  odiff[{y}] -= 1"))
    times, run = [], 0
    for i in range(m):
        run += od[i]
        times.append(run)
        steps.append(row_step(f"prefix sum: op {i + 1} runs {run} times",
                              [("odiff", od[:], i), ("times", _grown(times, i + 1), i)],
                              f"run += odiff[{i}] -> {run}"))
    return _pack("Greg and Array - pass 1, count the executions",
                 "ops = [(1,2,1), (1,3,2), (2,3,4)], queries = [(1,2), (1,3), (2,3)]",
                 steps, times)


def _greg_array():
    a = list(_GREG_A)
    times = [2, 3, 2]
    n, m = len(a), len(_GREG_OPS)
    ad = [0] * (n + 1)
    steps = [row_step("now the axis is array cells",
                      [("a", a, None), ("adiff", ad[:], None)],
                      "op i adds times[i] * d_i over its range")]
    for idx, ((l, r, dv), t) in enumerate(zip(_GREG_OPS, times)):
        ad[l - 1] += t * dv
        ad[r] -= t * dv
        steps.append(row_step(f"op {idx + 1}: [{l},{r}] + {t} x {dv}",
                              [("a", a, None), ("adiff", ad[:], l - 1)],
                              f"adiff[{l - 1}] += {t * dv},  adiff[{r}] -= {t * dv}"))
    out, run = [], 0
    for i in range(n):
        run += ad[i]
        out.append(a[i] + run)
    steps.append(row_step("one prefix sum finishes the array",
                          [("adiff", ad[:], None), ("out", out, None)],
                          "out[i] = a[i] + prefix(adiff)[i]"))
    return _pack("Greg and Array - pass 2, fold into the array",
                 "a = [1, 2, 3], times = [2, 3, 2]", steps, out)


def _mat2d_table():
    mat = [[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7],
           [1, 0, 3, 0, 5]]
    m, n = len(mat), len(mat[0])
    p = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        rowsum = 0
        for j in range(n):
            rowsum += mat[i][j]
            p[i + 1][j + 1] = p[i][j + 1] + rowsum
    blank = [["?"] * (n + 1) for _ in range(m + 1)]

    def partial(rows):
        grid = [row[:] for row in blank]
        for i in range(rows + 1):
            for j in range(n + 1):
                grid[i][j] = p[i][j]
        return grid

    steps = [grid_step("P[i][j] = block above + running row sum", partial(0),
                       hi=(1, 1), eq="row 0 of the table")]
    steps.append(grid_step("the finished table, with a zero border", partial(m),
                           hi=(m, n), eq=f"P[{m}][{n}] = {p[m][n]} = sum of everything"))
    for (r1, c1, r2, c2) in ((2, 1, 4, 3), (1, 1, 2, 2), (1, 2, 2, 4)):
        got = p[r2 + 1][c2 + 1] - p[r1][c2 + 1] - p[r2 + 1][c1] + p[r1][c1]
        grid = partial(m)
        marks = {(r2 + 1, c2 + 1): "green", (r1, c2 + 1): "red",
                 (r2 + 1, c1): "red", (r1, c1): "amber"}
        steps.append(grid_step(
            f"region rows {r1}..{r2}, cols {c1}..{c2}", grid, marks=marks,
            eq=f"{p[r2 + 1][c2 + 1]} - {p[r1][c2 + 1]} - {p[r2 + 1][c1]} + {p[r1][c1]} = {got}"))
    return _pack("Range Sum Query 2D - four reads",
                 "5 x 5 matrix, sumRegion(2, 1, 4, 3) first",
                 steps, [8, 11, 12])


def _prod_tables():
    a = [1, 2, 3, 4]
    n = len(a)
    L, R = [1] * n, [1] * n
    for i in range(1, n):
        L[i] = L[i - 1] * a[i - 1]
    for i in range(n - 2, -1, -1):
        R[i] = R[i + 1] * a[i + 1]
    out = [L[i] * R[i] for i in range(n)]
    steps = [row_step("forward pass: everything to the left",
                      [("a", a, None), ("L", _grown(L, 2), 1)],
                      f"L[1] = L[0] * a[0] = {L[1]}")]
    steps.append(row_step("keep going",
                          [("a", a, None), ("L", L, n - 1)],
                          f"L[3] = L[2] * a[2] = {L[3]}"))
    steps.append(row_step("backward pass: everything to the right",
                          [("a", a, 3), ("L", L, None), ("R", _grown(R, n - 1), n - 2)],
                          f"R[2] = R[3] * a[3] = {R[2]}"))
    steps.append(row_step("the two tables never touch a[i] itself",
                          [("a", a, None), ("L", L, None), ("R", R, 0)],
                          f"R[0] = R[1] * a[1] = {R[0]}"))
    steps.append(row_step("answer = left * right",
                          [("L", L, None), ("R", R, None), ("out", out, None)],
                          f"out[0] = {L[0]} * {R[0]} = {out[0]}"))
    return _pack("Product of Array Except Self - two tables", "a = [1, 2, 3, 4]", steps, out)


def _prod_one():
    a = [1, 2, 3, 4]
    n = len(a)
    out = [1] * n
    for i in range(1, n):
        out[i] = out[i - 1] * a[i - 1]
    steps = [row_step("the forward pass writes into the answer",
                      [("a", a, None), ("out", out, None)],
                      "out is the left-product table L")]
    run = 1
    for i in range(n - 2, -1, -1):
        run *= a[i + 1]
        out[i] *= run
        steps.append(row_step(f"backwards at i = {i}",
                              [("a", a, i + 1), ("out", out[:], i)],
                              f"run = {run},  out[{i}] *= run -> {out[i]}"))
    steps.append(row_step("one variable replaces the second table",
                          [("a", a, None), ("out", out[:], None)],
                          "out = [24, 12, 8, 6]"))
    return _pack("Product of Array Except Self - one array", "a = [1, 2, 3, 4]", steps,
                 [24, 12, 8, 6])


def _xorq_table():
    a = [1, 3, 4, 8]
    queries = [[0, 1], [1, 2], [0, 3], [3, 3]]
    p = [0]
    for x in a:
        p.append(p[-1] ^ x)
    steps = [row_step("XOR instead of +", [("a", a, None), ("P", _grown(p, 2), 1)],
                      f"P[1] = P[0] ^ a[0] = {p[1]}")]
    steps.append(row_step("the identity is still 0",
                          [("a", a, None), ("P", _grown(p, 4), 3)],
                          f"P[3] = P[2] ^ a[2] = {p[3]}"))
    steps.append(row_step("table complete", [("a", a, None), ("P", p, 4)],
                          f"P[4] = {p[4]} = xor of everything"))
    for l, r in queries:
        steps.append(row_step(f"query [{l}, {r}]", [("P", p, r + 1)],
                              f"P[{l}] ^ P[{r + 1}] = {p[l]} ^ {p[r + 1]} = {p[l] ^ p[r + 1]}"))
    ans = [p[l] ^ p[r + 1] for l, r in queries]
    return _pack("XOR Queries of a Subarray - self-inverse",
                 "a = [1, 3, 4, 8], queries = [[0,1], [1,2], [0,3], [3,3]]", steps, ans)


def _split2_scan():
    a = [10, 4, -8, 7]
    total, left, ans = sum(a), 0, 0
    steps = [chip_step("one running part, the other is total - left",
                       [("total", total, "muted"), ("left", 0, "indigo"), ("ans", 0, "muted")],
                       "left >= total - left")]
    for i in range(len(a) - 1):
        left += a[i]
        right = total - left
        ok = left >= right
        ans += 1 if ok else 0
        steps.append(chip_step(f"cut after index {i}",
                               [("left", left, "green" if ok else "indigo"),
                                ("right", right, "amber" if not ok else "green"),
                                ("ans", ans, "muted")],
                               f"{left} >= {right}: count" if ok else f"{left} < {right}: skip"))
    return _pack("Number of Ways to Split Array - three cuts", "a = [10, 4, -8, 7]", steps, ans)


def _split3_pointers():
    a = [1, 3, 1, 1, 4]
    n = len(a)
    p, total = _prefix(a), sum(a)
    best, j = 0, n
    steps = [row_step("two cuts, one table", [("P", p, n)],
                  f"right part is P[n] - P[j], j starts at {n}")]
    for i in range(4):
        left = p[i]
        moved = False
        while j > i and total - p[j] < left:
            j -= 1
            moved = True
        right = total - p[j] if j >= i else None
        hit = right == left
        best = max(best, left) if hit else best
        eq = (f"P[{i}] = {left}, right part = {right}" + ("   equal -> best = " + str(best)
              if hit else ""))
        steps.append(row_step(("j moved to " + str(j)) if moved else f"i = {i}, j = {j}",
                              [("P", p, i)], eq))
    steps.append(row_step("j never moves forward, so this is O(n)",
                          [("P", p, j)], f"best sum_1 = {best}"))
    return _pack("Three Parts of the Array - two pointers",
                 "d = [1, 3, 1, 1, 4]", steps, best)


_ALL = None


def get(problem, form):
    """Return ``(title, input line, steps, final)`` for one solution form."""
    global _ALL
    if _ALL is None:
        _ALL = {
            ("range", "table"): _range_table(),
            ("pivot", "table"): _pivot_table(),
            ("pivot", "scan"): _pivot_scan(),
            ("sumk", "map"): _sumk_map(),
            ("divk", "buckets"): _divk_buckets(),
            ("good", "prefix"): _good_prefix(),
            ("flight", "diff"): _flight_diff(),
            ("greg", "times"): _greg_times(),
            ("greg", "array"): _greg_array(),
            ("mat2d", "table"): _mat2d_table(),
            ("prod", "tables"): _prod_tables(),
            ("prod", "one"): _prod_one(),
            ("xorq", "table"): _xorq_table(),
            ("split2", "scan"): _split2_scan(),
            ("split3", "pointers"): _split3_pointers(),
        }
    return _ALL[(problem, form)]
