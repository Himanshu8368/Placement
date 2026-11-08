"""Step boards for the matrices & 2D-arrays chapter (traversals, transforms, marking,
fix-two-rows reduction, coordinate/parity invariants).

Drawing lives in ``dsa_viz.story``; ``get`` returns data only. Each story walks the
chapter's own sample, at most eight frames, and every number comes from running the
algorithm - never from a hand-typed table. A matrix frame shows a grid with the
just-emitted band / the current cell / the flagged header highlighted, so the reader
watches a spiral edge, a transpose pass, a marker write, or a row-collapse exactly as
the algorithm performs it.

Every problem carries a ladder: a ``brute``/first frame set (the direct definition),
and an ``optimal``/second frame set (the rung the pattern names). ``get`` returns
whichever frame set the caller asks for.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from trace_log import chip_step, grid_step, row_step, show  # noqa: E402


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    if not 1 <= len(steps) <= 8:
        raise AssertionError(f"{title}: {len(steps)} frames")
    return (title, inp, steps, text)


def _grid(M):
    return [[str(v) for v in row] for row in M]


def _diag_band(M, s):
    """marks for cells with i+j == s."""
    return {(i, j): "green" for i in range(len(M)) for j in range(len(M[0])) if i + j == s}


# ---------------------------------------------------------------- LC 54 spiral
SPIRAL = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
SPIRAL_ANS = [1, 2, 3, 6, 9, 8, 7, 4, 5]


def _spiral_visited():
    R, C = len(SPIRAL), len(SPIRAL[0])
    seen = [[False] * C for _ in range(R)]
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    r = c = di = 0
    out = []
    steps = [grid_step("input", _grid(SPIRAL),
                       eq=f"{R}x{C}; walk right/down/left/up, turn on wall or seen")]
    for _ in range(R * C):
        out.append(SPIRAL[r][c])
        seen[r][c] = True
        nr, nc = r + dirs[di][0], c + dirs[di][1]
        if not (0 <= nr < R and 0 <= nc < C and not seen[nr][nc]):
            di = (di + 1) % 4
            nr, nc = r + dirs[di][0], c + dirs[di][1]
        r, c = nr, nc
        if len(out) in (3, 5, 7, 9):        # show each turn
            marks = {(i, j): "green" for i in range(R) for j in range(C) if seen[i][j]}
            marks[(r, c)] = "amber"
            steps.append(grid_step(f"emit {len(out)}", _grid(SPIRAL), hi=(r, c),
                                   marks=marks, eq="next=" + str(out)))
    assert out == SPIRAL_ANS
    return _pack("Spiral - direction walk with a visited set",
                 _grid(SPIRAL)[0], steps, str(out))


def _spiral_walls():
    top, bottom, left, right = 0, 2, 0, 2
    out = []
    emitted = set()
    steps = [grid_step("input", _grid(SPIRAL),
                       eq="walls top=0 bottom=2 left=0 right=2")]
    edges = []
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            edges.append((top, j)); out.append(SPIRAL[top][j])
        top += 1
        for i in range(top, bottom + 1):
            edges.append((i, right)); out.append(SPIRAL[i][right])
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1):
                edges.append((bottom, j)); out.append(SPIRAL[bottom][j])
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                edges.append((i, left)); out.append(SPIRAL[i][left])
            left += 1
        marks = {(i, j): "green" for (i, j) in edges}
        steps.append(grid_step(f"lap walls t{top}-1 b{bottom} l{left} r{right}",
                               _grid(SPIRAL), marks=marks,
                               eq=f"walls top={top} bottom={bottom} left={left} right={right}"))
    assert out == SPIRAL_ANS
    return _pack("Spiral - shrinking walls (optimal)", _grid(SPIRAL)[0], steps, str(out))


# ------------------------------------------------------- LC 498 diagonal traverse
DIAGT = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
DIAGT_ANS = [1, 2, 4, 7, 5, 3, 6, 8, 9]


def _diag_bucket():
    R, C = len(DIAGT), len(DIAGT[0])
    buckets = [[] for _ in range(R + C - 1)]
    for i in range(R):
        for j in range(C):
            buckets[i + j].append((i, j))
    out = []
    steps = [grid_step("input", _grid(DIAGT), eq="group cells by the key s = i + j")]
    for s in range(R + C - 1):
        line = [DIAGT[i][j] for (i, j) in buckets[s]]
        take = line[::-1] if s % 2 == 0 else line
        out += take
        steps.append(grid_step(f"s={s} ({'up' if s % 2 == 0 else 'down'})",
                               _grid(DIAGT), marks=_diag_band(DIAGT, s),
                               eq=f"diagonal {line} -> append {take}; out={out}"))
    assert out == DIAGT_ANS
    return _pack("Diagonal traverse - i+j buckets zig-zag",
                 _grid(DIAGT)[0], steps, str(out))


def _diag_walk():
    R, C = len(DIAGT), len(DIAGT[0])
    r = c = 0
    up = True
    out = []
    walk = []                       # (emitted cell, running output copy)
    for _ in range(R * C):
        out.append(DIAGT[r][c])
        walk.append(((r, c), list(out)))
        if up:
            if c == C - 1:
                up = False; r += 1
            elif r == 0:
                up = False; c += 1
            else:
                r -= 1; c += 1
        else:
            if r == R - 1:
                up = True; c += 1
            elif c == 0:
                up = True; r += 1
            else:
                r += 1; c -= 1
    assert out == DIAGT_ANS
    # one frame per diagonal: the last cell emitted while on that i+j band
    steps = [grid_step("input", _grid(DIAGT), hi=(0, 0), marks={(0, 0): "green"},
                       eq="zig-zag: ride one diagonal, hop at the edge, flip direction")]
    last_on_band = {}
    for idx, ((er, ec), partial) in enumerate(walk):
        last_on_band[er + ec] = idx
    for s in range(R + C - 1):
        idx = last_on_band[s]
        (er, ec), partial = walk[idx]
        marks = {(a, b): "green" for a in range(R) for b in range(C) if (a + b) < s}
        marks[(er, ec)] = "amber"
        steps.append(grid_step(f"diagonal s={s} done", _grid(DIAGT), hi=(er, ec),
                               marks=marks, eq=f"out={partial}"))
    if len(steps) > 8:
        steps = [steps[0]] + steps[1:7] + [steps[-1]]
    return _pack("Diagonal traverse - O(1)-index pointer walk (optimal)",
                 _grid(DIAGT)[0], steps, str(out))


# --------------------------------------------------------------- LC 48 rotate image
ROT = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
ROT_ANS = [[7, 4, 1], [8, 5, 2], [9, 6, 3]]


def _transpose(m):
    n = len(m)
    return [[m[i][j] for i in range(n)] for j in range(n)]


def _rotate_extra():
    T = _transpose(ROT)
    Rv = [row[::-1] for row in T]
    steps = [
        grid_step("input", _grid(ROT), eq="90 CW = transpose, then reverse each row"),
        grid_step("transpose", _grid(T),
                  marks={(i, j): "cyan" for i in range(3) for j in range(3) if i != j},
                  eq="swap A[i][j] <-> A[j][i]"),
        grid_step("reverse rows", _grid(Rv),
                  marks={(i, j): "green" for i in range(3) for j in range(3)},
                  eq="reverse every row -> the answer"),
    ]
    assert Rv == ROT_ANS
    return _pack("Rotate image - transpose + reverse (a new array)",
                 _grid(ROT)[0], steps, str(ROT_ANS))


def _rotate_cycle():
    A = [row[:] for row in ROT]
    n = 3
    steps = [grid_step("input", _grid(A),
                       marks={(0, 0): "amber", (0, 2): "amber", (2, 2): "amber", (2, 0): "amber"},
                       eq="four-cycle the corners (0,0)->(0,2)->(2,2)->(2,0)")]
    i = 0
    for j in range(i, n - 1 - i):
        A[i][j], A[j][n - 1 - i], A[n - 1 - i][n - 1 - j], A[n - 1 - j][i] = \
            A[n - 1 - j][i], A[i][j], A[j][n - 1 - i], A[n - 1 - i][n - 1 - j]
        quad = [(i, j), (j, n - 1 - i), (n - 1 - i, n - 1 - j), (n - 1 - j, i)]
        steps.append(grid_step(f"cycle {(i, j)}", _grid(A),
                               marks={q: "green" for q in quad},
                               eq="rotate the 4-cycle clockwise in place"))
    assert A == ROT_ANS
    return _pack("Rotate image - four-cycle in place, O(1) space (optimal)",
                 _grid(ROT)[0], steps, str(ROT_ANS))


# ---------------------------------------------------------- LC 867 transpose matrix
TRANS = [[1, 2, 3], [4, 5, 6]]
TRANS_ANS = [[1, 4], [2, 5], [3, 6]]


def _transpose_nested():
    R, C = len(TRANS), len(TRANS[0])
    T = [[0] * R for _ in range(C)]
    for j in range(C):
        for i in range(R):
            T[j][i] = TRANS[i][j]
    steps = [grid_step("input (2x3)", _grid(TRANS), hi=(0, 0),
                       marks={(0, 0): "cyan"},
                       eq="T[j][i] = A[i][j]; output has C rows, R cols")]
    for j in range(C):
        marks = {(i, j): "green" for i in range(R)}
        steps.append(grid_step(f"col {j} -> row {j}", _grid(TRANS), marks=marks,
                               eq=f"T[{j}] = {[TRANS[i][j] for i in range(R)]}"))
    assert T == TRANS_ANS
    return _pack("Transpose - nested build into a C x R array",
                 _grid(TRANS)[0], steps, str(TRANS_ANS))


def _transpose_comprehension():
    T = [list(col) for col in zip(*TRANS)]
    steps = [
        grid_step("input", _grid(TRANS), eq="zip(*A) gathers each column"),
        grid_step("result", _grid(T),
                  marks={(i, j): "green" for i in range(3) for j in range(2)},
                  eq="each original column becomes a row"),
    ]
    assert T == TRANS_ANS
    return _pack("Transpose - column zip (optimal one-liner)",
                 _grid(TRANS)[0], steps, str(TRANS_ANS))


# ------------------------------------------------------- LC 73 set matrix zeroes
ZERO = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
ZERO_ANS = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]


def _zeroes_sets():
    R = C = 3
    rows = {i for i in range(R) for j in range(C) if ZERO[i][j] == 0}
    cols = {j for j in range(C) for i in range(R) if ZERO[i][j] == 0}
    res = [[0 if (i in rows or j in cols) else ZERO[i][j] for j in range(C)] for i in range(R)]
    steps = [
        grid_step("input", _grid(ZERO), marks={(1, 1): "red"}, eq="the zero at (1,1) dooms row 1 and col 1"),
        grid_step("record", _grid(ZERO), marks={(1, 1): "red"},
                  eq=f"rows={rows} cols={cols} (extra O(R+C))"),
        grid_step("apply", _grid(res),
                  marks={(1, j): "amber" for j in range(C)} | {(i, 1): "amber" for i in range(R)},
                  eq="zero every doomed row and column"),
    ]
    assert res == ZERO_ANS
    return _pack("Set zeroes - row/col boolean sets", _grid(ZERO)[0], steps, str(ZERO_ANS))


def _zeroes_mark():
    M = [row[:] for row in ZERO]
    R = C = 3
    zr = any(M[0][j] == 0 for j in range(C))
    zc = any(M[i][0] == 0 for i in range(R))
    steps = [grid_step("save corner flags", _grid(M),
                       marks={(0, 0): "violet"}, eq=f"zero_row0={zr}, zero_col0={zc}")]
    for i in range(1, R):
        for j in range(1, C):
            if M[i][j] == 0:
                M[i][0] = 0
                M[0][j] = 0
    steps.append(grid_step("pass 1 mark headers", _grid(M),
                           marks={(1, 0): "cyan", (0, 1): "cyan"},
                           eq="borrow row 0 / col 0 as the flag arrays"))
    for i in range(1, R):
        for j in range(1, C):
            if M[i][0] == 0 or M[0][j] == 0:
                M[i][j] = 0
    steps.append(grid_step("pass 2 wipe interior", _grid(M),
                           marks={(1, 2): "amber", (2, 1): "amber"},
                           eq="zero interior cells under a flagged header"))
    if zr:
        for j in range(C):
            M[0][j] = 0
    if zc:
        for i in range(R):
            M[i][0] = 0
    steps.append(grid_step("apply saved flags", _grid(M),
                           marks={(0, 1): "green", (1, 0): "green"},
                           eq="row 0 / col 0 last, from the two booleans"))
    assert M == ZERO_ANS
    return _pack("Set zeroes - O(1) header marking (optimal)",
                 _grid(ZERO)[0], steps, str(ZERO_ANS))


# -------------------------------------------- LC 1074 submatrices that sum to target
SUB = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
SUB_TARGET = 0
SUB_ANS = 4


def _sub_brute():
    R = C = 3
    cnt = 0
    steps = [grid_step("input", _grid(SUB),
                       eq=f"count rectangles with sum == {SUB_TARGET}: four nested loops")]
    for t in range(R):
        for b in range(t, R):
            for l in range(C):
                for r in range(l, C):
                    s = sum(SUB[i][j] for i in range(t, b + 1) for j in range(l, r + 1))
                    if s == SUB_TARGET:
                        cnt += 1
                        marks = {(i, j): "green" for i in range(t, b + 1) for j in range(l, r + 1)}
                        steps.append(grid_step(f"rect ({t},{l})-({b},{r})", _grid(SUB), marks=marks,
                                               eq=f"sum={s} == target -> count={cnt}"))
    steps = [steps[0]] + steps[1:6]
    assert cnt == SUB_ANS
    return _pack("Submatrix sum target - O(R^2 C^2) brute",
                 _grid(SUB)[0], steps, str(cnt))


def _sub_collapse():
    from collections import defaultdict
    R = C = 3
    total = 0
    steps = [grid_step("input", _grid(SUB),
                       eq="fix (top,bottom); collapse columns; count subarrays == target")]
    band_rows = []
    for t in range(R):
        col = [0] * C
        for b in range(t, R):
            col = [col[j] + SUB[b][j] for j in range(C)]
            cnt = defaultdict(int)
            pre = 0
            cnt[0] = 1
            for v in col:
                pre += v
                total += cnt[pre - SUB_TARGET]
                cnt[pre] += 1
            band_rows.append((t, b, list(col), total))
    for (t, b, col, tot) in band_rows[:5]:
        marks = {(b, j): "green" for j in range(C)}
        steps.append(grid_step(f"top={t} bottom={b}", _grid(SUB), marks=marks,
                               eq=f"col-sum={col}; running count={tot}"))
    assert total == SUB_ANS
    return _pack("Submatrix sum target - fix two rows + prefix hash (optimal)",
                 _grid(SUB)[0], steps, str(total))


# ---------------------------------------------------- LC 1572 matrix diagonal sum
DSUM = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
DSUM_ANS = 25


def _dsum_twopass():
    n = 3
    main = sum(DSUM[i][i] for i in range(n))
    anti = sum(DSUM[i][n - 1 - i] for i in range(n))
    overlap = DSUM[n // 2][n // 2] if n % 2 == 1 else 0
    ans = main + anti - overlap
    steps = [
        grid_step("main diagonal", _grid(DSUM),
                  marks={(i, i): "green" for i in range(n)}, eq=f"sum = {main}"),
        grid_step("anti diagonal", _grid(DSUM),
                  marks={(i, n - 1 - i): "cyan" for i in range(n)},
                  eq=f"sum = {anti}; centre {overlap} counted twice"),
        grid_step("answer", _grid(DSUM),
                  marks={(i, i): "green" for i in range(n)} |
                        {(i, n - 1 - i): "cyan" for i in range(n)},
                  eq=f"{main} + {anti} - {overlap} = {ans}"),
    ]
    assert ans == DSUM_ANS
    return _pack("Matrix diagonal sum - two passes, subtract centre",
                 _grid(DSUM)[0], steps, str(ans))


def _dsum_oneloop():
    n = 3
    s = 0
    steps = [grid_step("input", _grid(DSUM), eq="one loop: A[i][i] + A[i][n-1-i]")]
    for i in range(n):
        s += DSUM[i][i] + DSUM[i][n - 1 - i]
        marks = {(i, i): "green", (i, n - 1 - i): "cyan"}
        steps.append(grid_step(f"row {i}", _grid(DSUM), marks=marks,
                               eq=f"add {DSUM[i][i]} + {DSUM[i][n - 1 - i]}; running={s}"))
    if n % 2 == 1:
        s -= DSUM[n // 2][n // 2]
        steps.append(grid_step("undo centre", _grid(DSUM),
                               marks={(n // 2, n // 2): "violet"},
                               eq=f"odd n: subtract {DSUM[n // 2][n // 2]} -> {s}"))
    assert s == DSUM_ANS
    return _pack("Matrix diagonal sum - single loop (optimal)",
                 _grid(DSUM)[0], steps, str(s))


# ---------------------------------------------------- CF 263A beautiful matrix
BEAU = [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]]
BEAU_ANS = 3


def _beau_scan():
    pos = None
    for i in range(5):
        for j in range(5):
            if BEAU[i][j] == 1:
                pos = (i, j)
    ans = abs(pos[0] - 2) + abs(pos[1] - 2)
    steps = [
        grid_step("input", _grid(BEAU), marks={(2, 2): "violet"},
                  eq="24 zeros and one 1; the centre (2,2) is the target"),
        grid_step("locate the 1", _grid(BEAU), hi=pos, marks={pos: "amber", (2, 2): "violet"},
                  eq=f"the 1 sits at {pos}"),
        grid_step("Manhattan", _grid(BEAU), marks={pos: "amber", (2, 2): "green"},
                  eq=f"|{pos[0]}-2| + |{pos[1]}-2| = {ans}"),
    ]
    assert ans == BEAU_ANS
    return _pack("Beautiful matrix - scan for the 1, distance to centre",
                 "5x5 grid", steps, str(ans))


def _beau_onepass():
    ans = 0
    for i in range(5):
        for j in range(5):
            if BEAU[i][j] == 1:
                ans = abs(i - 2) + abs(j - 2)
    steps = [
        row_step("row scan", [("row i", [str(r) for r in range(5)], 1)],
                 eq="one pass: when you read the 1, answer |i-2|+|j-2| and stop"),
        grid_step("answer", _grid(BEAU), marks={(1, 4): "amber", (2, 2): "green"},
                  eq=f"found at (1,4) -> {ans}"),
    ]
    assert ans == BEAU_ANS
    return _pack("Beautiful matrix - single sweep (optimal)", "5x5 grid", steps, str(ans))


# ------------------------------------------------------------- contest-ready rungs
# A third, submission-shaped mechanic for each problem: different code, same answer.


def _spiral_ring():
    R, C = len(SPIRAL), len(SPIRAL[0])
    out = []
    steps = [grid_step("input", _grid(SPIRAL),
                       eq="peel rings: for k in range((min(R,C)+1)//2)")]
    for k in range((min(R, C) + 1) // 2):
        top, bottom, left, right = k, R - 1 - k, k, C - 1 - k
        for j in range(left, right + 1):
            out.append(SPIRAL[top][j])
        for i in range(top + 1, bottom + 1):
            out.append(SPIRAL[i][right])
        if top < bottom:
            for j in range(right - 1, left - 1, -1):
                out.append(SPIRAL[bottom][j])
        if left < right:
            for i in range(bottom - 1, top, -1):
                out.append(SPIRAL[i][left])
        ring = {(top, j) for j in range(left, right + 1)} | \
               {(i, right) for i in range(top, bottom + 1)} | \
               {(bottom, j) for j in range(left, right + 1)} | \
               {(i, left) for i in range(top, bottom + 1)}
        steps.append(grid_step(f"ring k={k}", _grid(SPIRAL), marks={q: "green" for q in ring},
                               eq=f"out={out}"))
    assert out == SPIRAL_ANS
    return _pack("Spiral - ring peel (contest)", _grid(SPIRAL)[0], steps, str(out))


def _diag_sweep():
    R, C = len(DIAGT), len(DIAGT[0])
    out = []
    steps = [grid_step("input", _grid(DIAGT),
                       eq="for band s: read rows i0..i1, reverse when s is even")]
    for s in range(R + C - 1):
        i0, i1 = max(0, s - (C - 1)), min(s, R - 1)
        band = [DIAGT[i][s - i] for i in range(i0, i1 + 1)]
        take = band[::-1] if s % 2 == 0 else band
        out += take
        marks = {(i, s - i): "green" for i in range(i0, i1 + 1)}
        steps.append(grid_step(f"band s={s}", _grid(DIAGT), marks=marks,
                               eq=f"band {band} -> {take}; out={out}"))
    assert out == DIAGT_ANS
    return _pack("Diagonal traverse - sweep each band by row range (contest)",
                 _grid(DIAGT)[0], steps, str(out))


def _rotate_formula():
    n = 3
    Rm = [[0] * n for _ in range(n)]
    steps = [grid_step("input", _grid(ROT),
                       eq="new[j][n-1-i] = A[i][j] into a fresh grid")]
    marks = {}
    for i in range(n):
        for j in range(n):
            Rm[j][n - 1 - i] = ROT[i][j]
            marks[(i, j)] = "cyan"
        steps.append(grid_step(f"scatter row {i}", _grid(ROT), marks=dict(marks),
                               eq=f"row {i} -> {[ROT[i][j] for j in range(n)][::-1]} placed as a column"))
        marks = {}
    steps.append(grid_step("result", _grid(Rm),
                           marks={(i, j): "green" for i in range(n) for j in range(n)},
                           eq=str(Rm)))
    assert Rm == ROT_ANS
    return _pack("Rotate image - direct scatter into a new grid (contest)",
                 _grid(ROT)[0], steps, str(ROT_ANS))


def _transpose_scatter():
    R, C = len(TRANS), len(TRANS[0])
    T = [[0] * R for _ in range(C)]
    steps = [grid_step("input (2x3)", _grid(TRANS),
                       eq="one loop over cells: T[j][i] = A[i][j]")]
    for i in range(R):
        marks = {(i, j): "green" for j in range(C)}
        for j in range(C):
            T[j][i] = TRANS[i][j]
        steps.append(grid_step(f"read row {i}", _grid(TRANS), marks=marks,
                               eq=f"{[TRANS[i][j] for j in range(C)]} -> down each output column {i}"))
    steps.append(grid_step("result (3x2)", _grid(T),
                           marks={(i, j): "cyan" for i in range(C) for j in range(R)},
                           eq=str(T)))
    assert T == TRANS_ANS
    return _pack("Transpose - cell scatter (contest)", _grid(TRANS)[0], steps, str(TRANS_ANS))


def _zeroes_copy():
    R = C = 3
    res = [row[:] for row in ZERO]
    zr = [False] * R
    zc = [False] * C
    for i in range(R):
        for j in range(C):
            if ZERO[i][j] == 0:
                zr[i] = True
                zc[j] = True
    steps = [grid_step("input", _grid(ZERO), marks={(1, 1): "red"},
                       eq=f"clone first; flags rows={zr} cols={zc}")]
    for i in range(R):
        for j in range(C):
            if zr[i] or zc[j]:
                res[i][j] = 0
    steps.append(grid_step("repaint clone", _grid(res),
                           marks={(1, j): "amber" for j in range(C)} | {(i, 1): "amber" for i in range(R)},
                           eq="never read a freshly-written 0 - the flags came from the original")
               )
    assert res == ZERO_ANS
    return _pack("Set zeroes - clone + boolean flags (contest-safe)",
                 _grid(ZERO)[0], steps, str(ZERO_ANS))


def _sub_prefix():
    R = C = 3
    P = [[0] * (C + 1) for _ in range(R + 1)]
    for i in range(R):
        for j in range(C):
            P[i + 1][j + 1] = SUB[i][j] + P[i][j + 1] + P[i + 1][j] - P[i][j]
    cnt = 0
    steps = [grid_step("integral image", [[str(P[i + 1][j + 1]) for j in range(C)] for i in range(R)],
                       eq="build prefix P; every rectangle sum is one O(1) query")]
    shown = 0
    for r1 in range(R):
        for r2 in range(r1, R):
            for c1 in range(C):
                for c2 in range(c1, C):
                    s = P[r2 + 1][c2 + 1] - P[r1][c2 + 1] - P[r2 + 1][c1] + P[r1][c1]
                    if s == SUB_TARGET:
                        cnt += 1
                        if shown < 4:
                            shown += 1
                            marks = {(i, j): "green" for i in range(r1, r2 + 1) for j in range(c1, c2 + 1)}
                            steps.append(grid_step(f"rect ({r1},{c1})-({r2},{c2})", _grid(SUB),
                                                   marks=marks,
                                                   eq=f"P query = {s} == target -> count={cnt}"))
    steps = steps[:6]
    assert cnt == SUB_ANS
    return _pack("Submatrix sum target - 2D prefix + four loops (contest)",
                 _grid(SUB)[0], steps, str(cnt))


def _dsum_single():
    n = 3
    s = 0
    steps = [grid_step("input", _grid(DSUM),
                       eq="one loop; add A[i][n-1-i] only when it is not A[i][i]")]
    for i in range(n):
        s += DSUM[i][i]
        marks = {(i, i): "green"}
        if n - 1 - i != i:
            s += DSUM[i][n - 1 - i]
            marks[(i, n - 1 - i)] = "cyan"
        else:
            marks[(i, i)] = "violet"
        steps.append(grid_step(f"row {i}", _grid(DSUM), marks=marks,
                               eq=f"running={s}" + ("  (centre counted once)" if n - 1 - i == i else "")))
    assert s == DSUM_ANS
    return _pack("Matrix diagonal sum - guarded single loop (contest)",
                 _grid(DSUM)[0], steps, str(s))


def _beau_stream():
    ans = 0
    steps = [row_step("read rows", [("row", [str(r) for r in range(5)], 1)],
                      eq="as each row streams in, a 1 immediately answers |i-2|+|j-2|")]
    for i in range(5):
        for j in range(5):
            if BEAU[i][j] == 1:
                ans = abs(i - 2) + abs(j - 2)
                steps.append(grid_step(f"read (1,{j})", _grid(BEAU), marks={(1, j): "amber", (2, 2): "green"},
                                       eq=f"the 1 arrives at row 1 col {j} -> {ans}"))
    assert ans == BEAU_ANS
    return _pack("Beautiful matrix - answer while reading input (contest)",
                 "5x5 grid", steps, str(ans))


# -------------------------------------------------------------------- registry
REGISTRY = {
    ("spiral", "visited"): _spiral_visited(),
    ("spiral", "walls"): _spiral_walls(),
    ("spiral", "ring"): _spiral_ring(),
    ("diagtraverse", "bucket"): _diag_bucket(),
    ("diagtraverse", "walk"): _diag_walk(),
    ("diagtraverse", "sweep"): _diag_sweep(),
    ("rotate", "extra"): _rotate_extra(),
    ("rotate", "cycle"): _rotate_cycle(),
    ("rotate", "formula"): _rotate_formula(),
    ("transpose", "nested"): _transpose_nested(),
    ("transpose", "zip"): _transpose_comprehension(),
    ("transpose", "scatter"): _transpose_scatter(),
    ("zeroes", "sets"): _zeroes_sets(),
    ("zeroes", "mark"): _zeroes_mark(),
    ("zeroes", "copy"): _zeroes_copy(),
    ("submatrix", "brute"): _sub_brute(),
    ("submatrix", "collapse"): _sub_collapse(),
    ("submatrix", "prefix"): _sub_prefix(),
    ("diagsum", "twopass"): _dsum_twopass(),
    ("diagsum", "oneloop"): _dsum_oneloop(),
    ("diagsum", "single"): _dsum_single(),
    ("beautiful", "scan"): _beau_scan(),
    ("beautiful", "onepass"): _beau_onepass(),
    ("beautiful", "stream"): _beau_stream(),
}


def get(problem, form):
    return REGISTRY[(problem, form)]
