"""Step boards for the back half of the grid DP chapter.

Patterns 3–5: backward and layered DP, two walkers, shape DP.
``stories`` returns ``(problem, form) -> (title, input line, steps, final)``.
Drawing lives in ``dsa_viz.story``. This module only records frames.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import CallLog, grid_step, line_step, row_step, run_call, show

MOD = 10**9 + 7
NEG = float("-inf")
POS = float("inf")

DUNGEON = [[-2, -3, 3], [-5, -10, 1], [10, 30, -5]]
CHERRY = [[0, 1, -1], [1, 0, -1], [1, 1, 1]]
CHERRY2 = [[3, 1, 1], [2, 5, 1], [1, 5, 5], [2, 1, 1]]
SQUARE = [
    ["1", "0", "1", "0", "0"],
    ["1", "0", "1", "1", "1"],
    ["1", "1", "1", "1", "1"],
    ["1", "0", "0", "1", "0"],
]
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    steps = list(steps)
    if len(steps) > 8:
        steps = steps[:7] + steps[-1:]
    return (title, inp, steps, text)


def _logged(build, force_collapse=False):
    if not force_collapse:
        return run_call(build)
    log = CallLog(collapse=True)
    answer = build(log)
    return answer, log.steps(8)


def _fmt_row(row):
    return [show(v) for v in row]


def _need_at(d, r, c, log, memo):
    """Least health on entering (r, c). Logged recursion, optional memo."""
    m, n = len(d), len(d[0])
    label = f"{r},{c}"
    if memo is not None and memo[r][c] is not None:
        return log.hit(label, memo[r][c])
    log.call(label)
    if r == m - 1 and c == n - 1:
        res = max(1, 1 - d[r][c])
        if memo is not None:
            memo[r][c] = res
        return log.base(res, f"{label} = {res}")
    best = None
    if r + 1 < m:
        best = _need_at(d, r + 1, c, log, memo)
    if c + 1 < n:
        right = _need_at(d, r, c + 1, log, memo)
        best = right if best is None else min(best, right)
    res = max(1, best - d[r][c])
    if memo is not None:
        memo[r][c] = res
    return log.ret(res, f"{label} = {res}")


def _dung_rec():
    def build(log):
        return _need_at(DUNGEON, 0, 0, log, None)

    ans, steps = _logged(build, force_collapse=True)
    return _pack(
        "Dungeon — recursion",
        "[[-2,-3,3],[-5,-10,1],[10,30,-5]]",
        steps,
        ans,
    )


def _dung_memo():
    def build(log):
        memo = [[None] * 3 for _ in range(3)]
        return _need_at(DUNGEON, 0, 0, log, memo)

    ans, steps = _logged(build, force_collapse=True)
    return _pack(
        "Dungeon — memo",
        "[[-2,-3,3],[-5,-10,1],[10,30,-5]]",
        steps,
        ans,
    )


def _dung_tab():
    d = DUNGEON
    m, n = len(d), len(d[0])
    need = [["·"] * n for _ in range(m)]
    num = [[POS] * (n + 1) for _ in range(m + 1)]
    num[m][n - 1] = 1
    num[m - 1][n] = 1
    steps = [grid_step("seed after the princess", [row[:] for row in need], eq="outside is ∞; the cell past her needs 1")]
    for r in range(m - 1, -1, -1):
        for c in range(n - 1, -1, -1):
            nxt = min(num[r + 1][c], num[r][c + 1])
            num[r][c] = max(1, nxt - d[r][c])
            need[r][c] = show(num[r][c])
            steps.append(grid_step(
                f"({r},{c})",
                [row[:] for row in need],
                hi=(r, c),
                eq=f"max(1, min(down, right) − ({d[r][c]})) = {show(num[r][c])}",
            ))
    return _pack(
        "Dungeon — table",
        "[[-2,-3,3],[-5,-10,1],[10,30,-5]]",
        steps,
        num[0][0],
    )


def _dung_row():
    d = DUNGEON
    m, n = len(d), len(d[0])
    row = [POS] * (n + 1)
    row[n - 1] = 1
    steps = [row_step("seed", [("row", _fmt_row(row), n - 1)], "row[n−1] = 1, row[n] stays ∞")]
    for r in range(m - 1, -1, -1):
        for c in range(n - 1, -1, -1):
            nxt = min(row[c], row[c + 1])
            row[c] = max(1, nxt - d[r][c])
        steps.append(row_step(
            f"after row {r}",
            [("row", _fmt_row(row), 0 if r == 0 else None)],
            "right neighbour is already new; below is still old",
        ))
    return _pack(
        "Dungeon — rolling row",
        "[[-2,-3,3],[-5,-10,1],[10,30,-5]]",
        steps,
        row[0],
    )


def _out_go(m, n, r, c, left, log, memo):
    inside = 0 <= r < m and 0 <= c < n
    label = f"{r},{c},{left}" if inside else "out"
    if inside and memo is not None and left < len(memo) and memo[left][r][c] is not None:
        return log.hit(label, memo[left][r][c])
    log.call(label)
    if not inside:
        return log.base(1, "out = 1")
    if left == 0:
        if memo is not None:
            memo[left][r][c] = 0
        return log.base(0, f"{label} = 0")
    res = 0
    for dr, dc in DIRS4:
        res += _out_go(m, n, r + dr, c + dc, left - 1, log, memo)
    res %= MOD
    if memo is not None:
        memo[left][r][c] = res
    return log.ret(res, f"{label} = {res}")


def _out_rec():
    def build(log):
        return _out_go(2, 2, 0, 0, 2, log, None)

    ans, steps = _logged(build, force_collapse=True)
    return _pack("Out of boundary — recursion", "(2, 2, 2, 0, 0)", steps, ans)


def _out_memo():
    def build(log):
        memo = [[[None] * 2 for _ in range(2)] for _ in range(3)]
        return _out_go(2, 2, 0, 0, 2, log, memo)

    ans, steps = _logged(build, force_collapse=True)
    return _pack("Out of boundary — memo", "(2, 2, 2, 0, 0)", steps, ans)


def _out_layers(roll):
    m = n = 2
    max_move = 2
    cur = [[0] * n for _ in range(m)]
    cur[0][0] = 1
    out = 0
    steps = [grid_step("layer 0", cur, hi=(0, 0), eq="one ball at the start")]
    for t in range(max_move):
        nxt = [[0] * n for _ in range(m)]
        for r in range(m):
            for c in range(n):
                if cur[r][c] == 0:
                    continue
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < m and 0 <= nc < n):
                        out = (out + cur[r][c]) % MOD
                    else:
                        nxt[nr][nc] = (nxt[nr][nc] + cur[r][c]) % MOD
        cur = nxt
        steps.append(grid_step(
            f"after move {t + 1}",
            [row[:] for row in cur],
            eq=f"exits so far = {out}",
        ))
    title = "Out of boundary — rolling layers" if roll else "Out of boundary — layers"
    return _pack(title, "(2, 2, 2, 0, 0)", steps, out)


def _out_tab():
    return _out_layers(False)


def _out_roll():
    return _out_layers(True)


def _cherry_f(g, t, r1, r2, log, memo):
    n = len(g)
    c1, c2 = t - r1, t - r2
    off = r1 < 0 or r2 < 0 or c1 < 0 or c2 < 0 or r1 >= n or r2 >= n or c1 >= n or c2 >= n
    thorn = (not off) and (g[r1][c1] < 0 or g[r2][c2] < 0)
    if off or thorn:
        label = "off" if off else "thorn"
        if log is not None:
            log.call(label)
            return log.base(NEG, label)
        return NEG
    if r1 > r2 and memo is not None:
        r1, r2 = r2, r1
    label = f"{t},{r1},{r2}"
    if memo is not None and (t, r1, r2) in memo:
        val = memo[(t, r1, r2)]
        log.hit(label, "−∞" if val == NEG else val)
        return val
    if log is not None:
        log.call(label)
    here = g[r1][t - r1] + (g[r2][t - r2] if r1 != r2 else 0)
    if t == 2 * n - 2:
        if memo is not None:
            memo[(t, r1, r2)] = here
        if log is not None:
            return log.base(here, f"{label} = {here}")
        return here
    best = NEG
    for d1 in (0, 1):
        for d2 in (0, 1):
            best = max(best, _cherry_f(g, t + 1, r1 + d1, r2 + d2, log, memo))
    res = NEG if best == NEG else here + best
    if memo is not None:
        memo[(t, r1, r2)] = res
    if log is None:
        return res
    shown = "−∞" if res == NEG else res
    return log.ret(res, f"{label} = {shown}")


def _cherry_public(g, log, memo):
    ans = _cherry_f(g, 0, 0, 0, log, memo)
    return 0 if ans == NEG else max(0, ans)


def _cherry_rec():
    def build(log):
        return _cherry_public(CHERRY, log, None)

    ans, steps = _logged(build, force_collapse=True)
    return _pack(
        "Cherry pickup — recursion",
        "[[0,1,-1],[1,0,-1],[1,1,1]]",
        steps,
        ans,
    )


def _cherry_memo():
    def build(log):
        return _cherry_public(CHERRY, log, {})

    ans, steps = _logged(build, force_collapse=True)
    return _pack(
        "Cherry pickup — memo",
        "[[0,1,-1],[1,0,-1],[1,1,1]]",
        steps,
        ans,
    )


def _cherry_grid(layer):
    n = len(layer)
    grid = []
    for r1 in range(n):
        row = []
        for r2 in range(n):
            if r1 > r2:
                row.append("·")
            elif layer[r1][r2] == NEG:
                row.append("X")
            else:
                row.append(show(layer[r1][r2]))
        grid.append(row)
    return grid


def _cherry_tab():
    g = CHERRY
    n = len(g)
    T = 2 * n - 1
    dp = [[[NEG] * n for _ in range(n)] for _ in range(T)]
    dp[0][0][0] = g[0][0]
    steps = [grid_step("t = 0", _cherry_grid(dp[0]), hi=(0, 0), eq=f"start = {g[0][0]}")]
    for t in range(1, T):
        lo, hi = max(0, t - n + 1), min(n - 1, t)
        for r1 in range(lo, hi + 1):
            for r2 in range(r1, hi + 1):
                c1, c2 = t - r1, t - r2
                if g[r1][c1] < 0 or g[r2][c2] < 0:
                    continue
                best = NEG
                for d1 in (0, 1):
                    for d2 in (0, 1):
                        p1, p2 = r1 - d1, r2 - d2
                        if p1 < 0 or p2 < 0:
                            continue
                        a, b = (p1, p2) if p1 <= p2 else (p2, p1)
                        best = max(best, dp[t - 1][a][b])
                if best == NEG:
                    continue
                dp[t][r1][r2] = best + g[r1][c1] + (g[r2][c2] if r1 != r2 else 0)
        steps.append(grid_step(f"t = {t}", _cherry_grid(dp[t]), hi=(n - 1, n - 1) if t == T - 1 else None))
    ans = dp[T - 1][n - 1][n - 1]
    return _pack(
        "Cherry pickup — table",
        "[[0,1,-1],[1,0,-1],[1,1,1]]",
        steps,
        0 if ans == NEG else max(0, ans),
    )


def _cherry_layer():
    g = CHERRY
    n = len(g)
    prev = [[NEG] * n for _ in range(n)]
    prev[0][0] = g[0][0]
    steps = [grid_step("t = 0", _cherry_grid(prev), hi=(0, 0), eq="one layer kept")]
    for t in range(1, 2 * n - 1):
        cur = [[NEG] * n for _ in range(n)]
        lo, hi = max(0, t - n + 1), min(n - 1, t)
        for r1 in range(lo, hi + 1):
            for r2 in range(r1, hi + 1):
                c1, c2 = t - r1, t - r2
                if g[r1][c1] < 0 or g[r2][c2] < 0:
                    continue
                best = prev[r1][r2]
                if r1 > 0:
                    best = max(best, prev[r1 - 1][r2])
                if r2 > r1:
                    best = max(best, prev[r1][r2 - 1])
                if r1 > 0:
                    best = max(best, prev[r1 - 1][r2 - 1])
                if best == NEG:
                    continue
                cur[r1][r2] = best + g[r1][c1] + (g[r2][c2] if r1 != r2 else 0)
        prev = cur
        steps.append(grid_step(f"t = {t}", _cherry_grid(prev), hi=(n - 1, n - 1) if t == 2 * n - 2 else None))
    ans = prev[n - 1][n - 1]
    return _pack(
        "Cherry pickup — rolling layer",
        "[[0,1,-1],[1,0,-1],[1,1,1]]",
        steps,
        0 if ans == NEG else max(0, ans),
    )


def _cherry2_f(g, r, a, b, log, memo):
    R, C = len(g), len(g[0])
    label = f"{r},{a},{b}"
    if memo is not None and memo[r][a][b] is not None:
        return log.hit(label, memo[r][a][b])
    if log is not None:
        log.call(label)
    here = g[r][a] + (g[r][b] if a != b else 0)
    if r == R - 1:
        if memo is not None:
            memo[r][a][b] = here
        if log is not None:
            return log.base(here, f"{label} = {here}")
        return here
    best = 0
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            na, nb = a + da, b + db
            if 0 <= na < C and 0 <= nb < C:
                best = max(best, _cherry2_f(g, r + 1, na, nb, log, memo))
    res = here + best
    if memo is not None:
        memo[r][a][b] = res
    if log is None:
        return res
    return log.ret(res, f"{label} = {res}")


def _cherry2_rec():
    def build(log):
        return _cherry2_f(CHERRY2, 0, 0, len(CHERRY2[0]) - 1, log, None)

    ans, steps = _logged(build, force_collapse=True)
    return _pack(
        "Cherry pickup II — recursion",
        "[[3,1,1],[2,5,1],[1,5,5],[2,1,1]]",
        steps,
        ans,
    )


def _cherry2_memo():
    def build(log):
        R, C = len(CHERRY2), len(CHERRY2[0])
        memo = [[[None] * C for _ in range(C)] for _ in range(R)]
        return _cherry2_f(CHERRY2, 0, 0, C - 1, log, memo)

    ans, steps = _logged(build, force_collapse=True)
    return _pack(
        "Cherry pickup II — memo",
        "[[3,1,1],[2,5,1],[1,5,5],[2,1,1]]",
        steps,
        ans,
    )


def _cherry2_tab():
    g = CHERRY2
    R, C = len(g), len(g[0])
    dp = [[[0] * C for _ in range(C)] for _ in range(R)]
    steps = []
    for r in range(R - 1, -1, -1):
        for a in range(C):
            for b in range(C):
                best = 0
                if r + 1 < R:
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            na, nb = a + da, b + db
                            if 0 <= na < C and 0 <= nb < C:
                                best = max(best, dp[r + 1][na][nb])
                dp[r][a][b] = best + g[r][a] + (g[r][b] if a != b else 0)
        steps.append(grid_step(
            f"row {r}",
            [row[:] for row in dp[r]],
            hi=(0, C - 1) if r == 0 else None,
            eq="dp[0][0][cols−1]" if r == 0 else "suffix, current cells included",
        ))
    return _pack(
        "Cherry pickup II — table",
        "[[3,1,1],[2,5,1],[1,5,5],[2,1,1]]",
        steps,
        dp[0][0][C - 1],
    )


def _cherry2_roll():
    g = CHERRY2
    R, C = len(g), len(g[0])
    below = [[0] * C for _ in range(C)]
    steps = []
    for r in range(R - 1, -1, -1):
        cur = [[0] * C for _ in range(C)]
        for a in range(C):
            for b in range(a, C):
                best = 0
                if r + 1 < R:
                    for na in range(max(0, a - 1), min(C - 1, a + 1) + 1):
                        for nb in range(max(0, b - 1), min(C - 1, b + 1) + 1):
                            best = max(best, below[min(na, nb)][max(na, nb)])
                cur[a][b] = best + g[r][a] + (g[r][b] if a != b else 0)
        below = cur
        grid = _cherry_grid(below)
        steps.append(grid_step(
            f"row {r}",
            grid,
            hi=(0, C - 1) if r == 0 else None,
            eq="answer is below[0][cols−1], not the max of the layer" if r == 0 else "only a ≤ b is stored",
        ))
    return _pack(
        "Cherry pickup II — rolling layer",
        "[[3,1,1],[2,5,1],[1,5,5],[2,1,1]]",
        steps,
        below[0][C - 1],
    )


def _square_f(g, r, c, log, memo):
    label = f"{r},{c}"
    if memo is not None and memo[r][c] is not None:
        return log.hit(label, memo[r][c])
    if log is not None:
        log.call(label)
    if g[r][c] == "0":
        res = 0
        if memo is not None:
            memo[r][c] = res
        if log is not None:
            return log.base(0, f"{label} = 0")
        return 0
    if r == 0 or c == 0:
        if memo is not None:
            memo[r][c] = 1
        if log is not None:
            return log.base(1, f"{label} = 1")
        return 1
    res = 1 + min(
        _square_f(g, r - 1, c, log, memo),
        _square_f(g, r, c - 1, log, memo),
        _square_f(g, r - 1, c - 1, log, memo),
    )
    if memo is not None:
        memo[r][c] = res
    if log is not None:
        return log.ret(res, f"{label} = {res}")
    return res


def _square_over(g, log, memo):
    best = 0
    if log is not None:
        log.call("area")
    for r in range(len(g)):
        for c in range(len(g[0])):
            best = max(best, _square_f(g, r, c, log, memo))
    ans = best * best
    if log is not None:
        return log.ret(ans, f"area = {ans}")
    return ans


def _square_rec():
    def build(log):
        return _square_over(SQUARE, log, None)

    ans, steps = _logged(build, force_collapse=True)
    return _pack(
        "Maximal square — recursion",
        '[["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]',
        steps,
        ans,
    )


def _square_memo():
    def build(log):
        memo = [[None] * len(SQUARE[0]) for _ in range(len(SQUARE))]
        return _square_over(SQUARE, log, memo)

    ans, steps = _logged(build, force_collapse=True)
    return _pack(
        "Maximal square — memo",
        '[["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]',
        steps,
        ans,
    )


def _square_tab():
    g = SQUARE
    m, n = len(g), len(g[0])
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best = 0
    steps = []
    for r in range(1, m + 1):
        for c in range(1, n + 1):
            if g[r - 1][c - 1] == "1":
                dp[r][c] = 1 + min(dp[r - 1][c], dp[r][c - 1], dp[r - 1][c - 1])
                best = max(best, dp[r][c])
        shown = [[show(dp[r][c]) for c in range(1, n + 1)] for r in range(1, m + 1)]
        # cells not yet filled stay blank in the display of later rows: rebuild from dp only up to this row
        view = []
        for i in range(1, m + 1):
            if i <= r:
                view.append([show(dp[i][c]) for c in range(1, n + 1)])
            else:
                view.append(["·"] * n)
        steps.append(grid_step(f"after input row {r - 1}", view, eq=f"best side {best}, area {best * best}"))
    return _pack(
        "Maximal square — table",
        '[["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]',
        steps,
        best * best,
    )


def _square_roll():
    g = SQUARE
    m, n = len(g), len(g[0])
    row = [0] * (n + 1)
    best = 0
    steps = []
    for r in range(1, m + 1):
        prev = 0
        for c in range(1, n + 1):
            up = row[c]
            if g[r - 1][c - 1] == "1":
                row[c] = 1 + min(up, row[c - 1], prev)
                best = max(best, row[c])
            else:
                row[c] = 0
            prev = up
        steps.append(row_step(
            f"row {r - 1}",
            [("row", _fmt_row(row), None)],
            f"prev saved the overwritten diagonal; area {best * best}",
        ))
    return _pack(
        "Maximal square — rolling row",
        '[["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]',
        steps,
        best * best,
    )


def _ones(matrix, r1, c1, r2, c2):
    for i in range(r1, r2 + 1):
        for j in range(c1, c2 + 1):
            if matrix[i][j] != "1":
                return False
    return True


def _rect_brute():
    g = SQUARE
    m, n = len(g), len(g[0])
    best = 0
    steps = []
    for r1 in range(m):
        for c1 in range(n):
            for r2 in range(r1, m):
                for c2 in range(c1, n):
                    if not _ones(g, r1, c1, r2, c2):
                        continue
                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    if area > best:
                        best = area
                        steps.append(line_step(
                            f"rows {r1}..{r2}, cols {c1}..{c2}",
                            [f"{r2 - r1 + 1}×{c2 - c1 + 1} of ones"],
                            f"best = {best}",
                        ))
    return _pack("Maximal rectangle — brute", "same matrix as maximal square", steps, best)


def _rect_prefix():
    g = SQUARE
    m, n = len(g), len(g[0])
    P = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            P[i + 1][j + 1] = (g[i][j] == "1") + P[i][j + 1] + P[i + 1][j] - P[i][j]
    best = 0
    steps = [line_step("prefix", ["P counts ones"], "a rectangle is solid iff ones = height × width")]
    for r1 in range(m):
        for c1 in range(n):
            for r2 in range(r1, m):
                for c2 in range(c1, n):
                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    ones = P[r2 + 1][c2 + 1] - P[r1][c2 + 1] - P[r2 + 1][c1] + P[r1][c1]
                    if ones != area:
                        break
                    if area > best:
                        best = area
                        steps.append(line_step(
                            f"rows {r1}..{r2}, cols {c1}..{c2}",
                            [f"ones = {ones} = area"],
                            f"best = {best}",
                        ))
    return _pack("Maximal rectangle — prefix check", "same matrix as maximal square", steps, best)


def _rect_stack():
    g = SQUARE
    m, n = len(g), len(g[0])
    h = [[0] * n for _ in range(m)]
    best = 0
    steps = []
    for r in range(m):
        for c in range(n):
            if g[r][c] == "1":
                h[r][c] = (h[r - 1][c] if r else 0) + 1
        st = []
        bars = h[r] + [0]
        for i, cur in enumerate(bars):
            while st and h[r][st[-1]] >= cur:
                top = st.pop()
                left = st[-1] if st else -1
                area = h[r][top] * (i - left - 1)
                if area > best:
                    best = area
                    steps.append(line_step(
                        f"row {r}, i = {i}",
                        [f"pop {top}: height {h[r][top]} × width {i - left - 1}"],
                        f"best = {best}",
                    ))
            if i < n:
                st.append(i)
    return _pack("Maximal rectangle — histogram stack", "same matrix as maximal square", steps, best)


def _rect_roll():
    g = SQUARE
    n = len(g[0])
    h = [0] * (n + 1)
    best = 0
    steps = []
    for r, row in enumerate(g):
        for c in range(n):
            h[c] = h[c] + 1 if row[c] == "1" else 0
        st = []
        row_best = 0
        for i in range(n + 1):
            while st and h[st[-1]] >= h[i]:
                top = st.pop()
                left = st[-1] if st else -1
                row_best = max(row_best, h[top] * (i - left - 1))
            st.append(i)
        best = max(best, row_best)
        steps.append(row_step(
            f"row {r}",
            [("h", _fmt_row(h), None)],
            f"sentinel h[{n}] = 0 closes the row; best = {best}",
        ))
    return _pack("Maximal rectangle — rolling heights", "same matrix as maximal square", steps, best)


def stories():
    """Every solved form in the back half of chapter 231."""
    return {
        ("dung", "rec"): _dung_rec(),
        ("dung", "memo"): _dung_memo(),
        ("dung", "tab"): _dung_tab(),
        ("dung", "row"): _dung_row(),
        ("out", "rec"): _out_rec(),
        ("out", "memo"): _out_memo(),
        ("out", "tab"): _out_tab(),
        ("out", "roll"): _out_roll(),
        ("cherry", "rec"): _cherry_rec(),
        ("cherry", "memo"): _cherry_memo(),
        ("cherry", "tab"): _cherry_tab(),
        ("cherry", "layer"): _cherry_layer(),
        ("cherry2", "rec"): _cherry2_rec(),
        ("cherry2", "memo"): _cherry2_memo(),
        ("cherry2", "tab"): _cherry2_tab(),
        ("cherry2", "roll"): _cherry2_roll(),
        ("square", "rec"): _square_rec(),
        ("square", "memo"): _square_memo(),
        ("square", "tab"): _square_tab(),
        ("square", "roll"): _square_roll(),
        ("rect", "brute"): _rect_brute(),
        ("rect", "prefix"): _rect_prefix(),
        ("rect", "stack"): _rect_stack(),
        ("rect", "roll"): _rect_roll(),
    }


if __name__ == "__main__":
    data = stories()
    for key in data:
        title, inp, steps, final = data[key]
        assert 1 <= len(steps) <= 8, (key, len(steps))
        print(f"{key[0]:8} {key[1]:6} steps={len(steps):2} final={final!r}  {title}")
