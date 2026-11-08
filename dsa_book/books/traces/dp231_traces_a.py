"""Step frames for the front half of two-dimensional grid DP.

Drawing lives elsewhere. ``stories`` returns data only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import chip_step, grid_step, line_step, row_step, run_call, show

MOD = 10**9 + 7
QUEEN = ["...", ".#.", "..."]
WORK = [[100, 100, 100], [100, 1, 100], [100, 100, 100]]
FALL = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
SCORE = ["E23", "2X2", "12S"]
PAREN = ["(((", ")()", "(()", "(()"]


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    if not 1 <= len(steps) <= 8:
        raise AssertionError((title, len(steps)))
    return (title, inp, steps, text)


def _queen_tables(g):
    H, W = len(g), len(g[0])
    dp = [[0] * W for _ in range(H)]
    X = [[0] * W for _ in range(H)]
    Y = [[0] * W for _ in range(H)]
    Z = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if g[r][c] == "#":
                continue
            v = 1 if r == 0 and c == 0 else 0
            if c:
                v += X[r][c - 1]
            if r:
                v += Y[r - 1][c]
            if r and c:
                v += Z[r - 1][c - 1]
            dp[r][c] = v % MOD
            X[r][c] = ((X[r][c - 1] if c else 0) + dp[r][c]) % MOD
            Y[r][c] = ((Y[r - 1][c] if r else 0) + dp[r][c]) % MOD
            Z[r][c] = ((Z[r - 1][c - 1] if r and c else 0) + dp[r][c]) % MOD
    return dp, X, Y, Z


def _marks_walls(g):
    return {(r, c): "red" for r, row in enumerate(g) for c, ch in enumerate(row) if ch == "#"}


def _queen_rec():
    def build(log):
        g = QUEEN

        def f(r, c):
            log.call(f"f({r},{c})")
            if r == 0 and c == 0:
                return log.base(1, "f(0,0) = 1")
            total = 0
            for dr, dc in ((0, -1), (-1, 0), (-1, -1)):
                k = 1
                while True:
                    pr, pc = r + dr * k, c + dc * k
                    if pr < 0 or pc < 0 or g[pr][pc] == "#":
                        break
                    total += f(pr, pc)
                    k += 1
            return log.ret(total % MOD, f"f({r},{c}) = {show(total % MOD)}")

        return f(2, 2)

    ans, steps = run_call(build)
    assert ans == 10
    return _pack("Queen on grid — recursion", '["...", ".#.", "..."]', steps, ans)


def _queen_memo():
    def build(log):
        g = QUEEN
        H, W = 3, 3
        memo = [[None] * W for _ in range(H)]

        def f(r, c):
            if memo[r][c] is not None:
                return log.hit(f"f({r},{c})", memo[r][c])
            log.call(f"f({r},{c})")
            if r == 0 and c == 0:
                memo[r][c] = 1
                return log.base(1, "f(0,0) = 1")
            total = 0
            for dr, dc in ((0, -1), (-1, 0), (-1, -1)):
                k = 1
                while True:
                    pr, pc = r + dr * k, c + dc * k
                    if pr < 0 or pc < 0 or g[pr][pc] == "#":
                        break
                    total += f(pr, pc)
                    k += 1
            memo[r][c] = total % MOD
            return log.ret(memo[r][c], f"f({r},{c}) = {show(memo[r][c])}")

        return f(2, 2)

    ans, steps = run_call(build)
    assert ans == 10
    return _pack("Queen on grid — memo", '["...", ".#.", "..."]', steps, ans)


def _queen_tab():
    g = QUEEN
    dp, _, _, _ = _queen_tables(g)
    assert dp[2][2] == 10
    marks = _marks_walls(g)
    shown = [["·"] * 3 for _ in range(3)]
    steps = []
    snap = {(0, 0), (0, 2), (1, 1), (1, 2), (2, 0), (2, 2)}
    eqs = {
        (0, 0): "the empty journey",
        (0, 2): "left ray 1 + 1",
        (1, 1): "a wall resets every ray",
        (1, 2): "0 + 2 + 1, the diagonal starts on the wall",
        (2, 0): "the up ray is the column sum 2",
        (2, 2): "5 + 5 + 0 = 10",
    }
    for r in range(3):
        for c in range(3):
            shown[r][c] = dp[r][c]
            if (r, c) in snap:
                steps.append(grid_step(
                    f"({r},{c}) → {show(dp[r][c])}",
                    [row[:] for row in shown],
                    hi=(r, c),
                    marks=marks,
                    eq=eqs[(r, c)],
                ))
    assert len(steps) <= 8
    return _pack("Queen on grid — table", '["...", ".#.", "..."]', steps, dp[2][2])


def _queen_roll():
    g = QUEEN
    H, W = 3, 3
    Y = [0] * W
    Z = [0] * (W + 1)
    steps = []
    ans = 0
    watch = {(0, 0), (0, 2), (1, 1), (1, 2), (2, 2)}
    for r in range(H):
        x = 0
        nZ = [0] * (W + 1)
        for c in range(W):
            if g[r][c] == "#":
                x = 0
                Y[c] = 0
                nZ[c + 1] = 0
                if (r, c) in watch:
                    steps.append(chip_step(
                        f"wall ({r},{c})",
                        [("x", 0, "red"), ("Y", 0, "red"), ("Z", 0, "red")],
                        "all three running sums reset",
                    ))
                continue
            v = 1 if r == 0 and c == 0 else 0
            v += x + Y[c] + Z[c]
            d = v % MOD
            prev_x, prev_y, prev_z = x, Y[c], Z[c]
            x = (x + d) % MOD
            Y[c] = (Y[c] + d) % MOD
            nZ[c + 1] = (Z[c] + d) % MOD
            if r == H - 1 and c == W - 1:
                ans = d
            if (r, c) in watch:
                steps.append(chip_step(
                    f"({r},{c}) → {show(d)}",
                    [("x", prev_x, "indigo"), ("Y", prev_y, "amber"), ("Z", prev_z, "violet"), ("dp", d, "green")],
                    f"{show(d)} = {show(prev_x)} + {show(prev_y)} + {show(prev_z)}" + (" + start" if (r, c) == (0, 0) else ""),
                ))
        Z = nZ
    assert ans == 10 and len(steps) <= 8
    return _pack("Queen on grid — rolling rays", '["...", ".#.", "..."]', steps, ans)


def _work_fns(log, memo=False):
    a = WORK
    n = m = 3
    cache = {}

    def call(name, i, j, body):
        key = (name, i, j)
        if memo and key in cache:
            return log.hit(f"{name}({i},{j})", cache[key])
        log.call(f"{name}({i},{j})")
        val = body()
        if memo:
            cache[key] = val
        return val

    def d1(i, j):
        def body():
            if i == 0 and j == 0:
                return log.base(a[0][0], "d1(0,0) = 100")
            best = 0
            if i:
                best = max(best, d1(i - 1, j))
            if j:
                best = max(best, d1(i, j - 1))
            return log.ret(a[i][j] + best, f"d1({i},{j}) = {show(a[i][j] + best)}")
        return call("d1", i, j, body)

    def d2(i, j):
        def body():
            if i == n - 1 and j == m - 1:
                return log.base(a[i][j], "d2 corner = 100")
            best = 0
            if i + 1 < n:
                best = max(best, d2(i + 1, j))
            if j + 1 < m:
                best = max(best, d2(i, j + 1))
            return log.ret(a[i][j] + best, f"d2({i},{j}) = {show(a[i][j] + best)}")
        return call("d2", i, j, body)

    def d3(i, j):
        def body():
            if i == n - 1 and j == 0:
                return log.base(a[i][j], "d3 corner = 100")
            best = 0
            if i + 1 < n:
                best = max(best, d3(i + 1, j))
            if j:
                best = max(best, d3(i, j - 1))
            return log.ret(a[i][j] + best, f"d3({i},{j}) = {show(a[i][j] + best)}")
        return call("d3", i, j, body)

    def d4(i, j):
        def body():
            if i == 0 and j == m - 1:
                return log.base(a[i][j], "d4 corner = 100")
            best = 0
            if i:
                best = max(best, d4(i - 1, j))
            if j + 1 < m:
                best = max(best, d4(i, j + 1))
            return log.ret(a[i][j] + best, f"d4({i},{j}) = {show(a[i][j] + best)}")
        return call("d4", i, j, body)

    return d1, d2, d3, d4


def _work_rec():
    def build(log):
        d1, d2, d3, d4 = _work_fns(log, memo=False)
        log.call("meet(1,1)")
        A = d1(1, 0) + d2(1, 2) + d3(2, 1) + d4(0, 1)
        B = d1(0, 1) + d2(2, 1) + d3(1, 0) + d4(1, 2)
        ans = max(A, B)
        return log.ret(ans, f"meet = {show(ans)}")

    ans, steps = run_call(build)
    assert ans == 800
    return _pack("Working out — recursion", "3×3, centre 1, else 100", steps, ans)


def _work_memo():
    def build(log):
        d1, d2, d3, d4 = _work_fns(log, memo=True)
        log.call("meet(1,1)")
        A = d1(1, 0) + d2(1, 2) + d3(2, 1) + d4(0, 1)
        B = d1(0, 1) + d2(2, 1) + d3(1, 0) + d4(1, 2)
        ans = max(A, B)
        return log.ret(ans, f"meet = {show(ans)}")

    ans, steps = run_call(build)
    assert ans == 800
    return _pack("Working out — memo", "3×3, centre 1, else 100", steps, ans)


def _corner_tables(a):
    n, m = len(a), len(a[0])
    d1 = [[0] * m for _ in range(n)]
    d2 = [[0] * m for _ in range(n)]
    d3 = [[0] * m for _ in range(n)]
    d4 = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            best = 0 if (i, j) == (0, 0) else -(10**18)
            if i:
                best = max(best, d1[i - 1][j])
            if j:
                best = max(best, d1[i][j - 1])
            if (i, j) == (0, 0):
                best = 0
            d1[i][j] = a[i][j] + best
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best = 0 if (i, j) == (n - 1, m - 1) else -(10**18)
            if i + 1 < n:
                best = max(best, d2[i + 1][j])
            if j + 1 < m:
                best = max(best, d2[i][j + 1])
            if (i, j) == (n - 1, m - 1):
                best = 0
            d2[i][j] = a[i][j] + best
    for i in range(n - 1, -1, -1):
        for j in range(m):
            best = 0 if (i, j) == (n - 1, 0) else -(10**18)
            if i + 1 < n:
                best = max(best, d3[i + 1][j])
            if j:
                best = max(best, d3[i][j - 1])
            if (i, j) == (n - 1, 0):
                best = 0
            d3[i][j] = a[i][j] + best
    for i in range(n):
        for j in range(m - 1, -1, -1):
            best = 0 if (i, j) == (0, m - 1) else -(10**18)
            if i:
                best = max(best, d4[i - 1][j])
            if j + 1 < m:
                best = max(best, d4[i][j + 1])
            if (i, j) == (0, m - 1):
                best = 0
            d4[i][j] = a[i][j] + best
    return d1, d2, d3, d4


def _work_tab():
    d1, d2, d3, d4 = _corner_tables(WORK)
    A = d1[1][0] + d2[1][2] + d3[2][1] + d4[0][1]
    B = d1[0][1] + d2[2][1] + d3[1][0] + d4[1][2]
    assert A == B == 800
    cells = [[show(v) for v in row] for row in WORK]
    steps = [
        grid_step("centre is the only interior cell", cells, hi=(1, 1), marks={(1, 1): "amber"},
                  eq="the 1 is not added"),
        chip_step("crossing A, neighbours",
                  [("d1 left", d1[1][0], "indigo"), ("d2 right", d2[1][2], "indigo"),
                   ("d3 below", d3[2][1], "amber"), ("d4 above", d4[0][1], "amber")],
                  "200 + 200 + 200 + 200"),
        chip_step("crossing B, neighbours",
                  [("d1 above", d1[0][1], "indigo"), ("d2 below", d2[2][1], "indigo"),
                   ("d3 left", d3[1][0], "amber"), ("d4 right", d4[1][2], "amber")],
                  "200 + 200 + 200 + 200"),
        line_step("both crossings", ["800 = max(800, 800)"], "eight cells of 100; the centre is skipped"),
    ]
    return _pack("Working out — four tables", "3×3, centre 1, else 100", steps, 800)


def _work_meet():
    a = WORK
    n, m = 3, 3
    d2 = [[0] * m for _ in range(n)]
    d3 = [[0] * m for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best = 0 if (i, j) == (n - 1, m - 1) else -(10**18)
            if i + 1 < n:
                best = max(best, d2[i + 1][j])
            if j + 1 < m:
                best = max(best, d2[i][j + 1])
            if (i, j) == (n - 1, m - 1):
                best = 0
            d2[i][j] = a[i][j] + best
        for j in range(m):
            best = 0 if (i, j) == (n - 1, 0) else -(10**18)
            if i + 1 < n:
                best = max(best, d3[i + 1][j])
            if j:
                best = max(best, d3[i][j - 1])
            if (i, j) == (n - 1, 0):
                best = 0
            d3[i][j] = a[i][j] + best
    p1 = [0] * m
    p4 = [0] * m
    steps = [line_step("d2 and d3 are finished", ["bottom-up tables stay in full"], "the sweep runs the other way")]
    best = 0
    for i in range(n):
        c1 = [0] * m
        c4 = [0] * m
        for j in range(m):
            prev = 0 if (i, j) == (0, 0) else -(10**18)
            if i:
                prev = max(prev, p1[j])
            if j:
                prev = max(prev, c1[j - 1])
            if (i, j) == (0, 0):
                prev = 0
            c1[j] = a[i][j] + prev
        for j in range(m - 1, -1, -1):
            prev = 0 if (i, j) == (0, m - 1) else -(10**18)
            if i:
                prev = max(prev, p4[j])
            if j + 1 < m:
                prev = max(prev, c4[j + 1])
            if (i, j) == (0, m - 1):
                prev = 0
            c4[j] = a[i][j] + prev
        if 0 < i < n - 1:
            for j in range(1, m - 1):
                A = c1[j - 1] + d2[i][j + 1] + d3[i + 1][j] + p4[j]
                B = p1[j] + d2[i + 1][j] + d3[i][j - 1] + c4[j + 1]
                best = max(best, A, B)
                steps.append(chip_step(
                    f"row {i}",
                    [("A", A, "indigo"), ("B", B, "amber"), ("best", best, "green")],
                    "d1 and d4 are the two live rows",
                ))
        p1, p4 = c1, c4
    assert best == 800 and len(steps) <= 8
    return _pack("Working out — rolled rows", "3×3, centre 1, else 100", steps, best)


def _fall_rec():
    def build(log):
        g, n = FALL, 3

        def f(r, c):
            log.call(f"f({r},{c})")
            if r == n - 1:
                return log.base(g[r][c], f"f({r},{c}) = {show(g[r][c])}")
            best, src = 10**18, 0
            for d in range(n):
                if d == c:
                    continue
                cand = f(r + 1, d)
                if cand < best:
                    best, src = cand, d
            return log.ret(g[r][c] + best, f"f({r},{c}) = {show(g[r][c])} + f({r + 1},{src})")

        log.call("min over row 0")
        ans, how = 10**18, 0
        for c in range(n):
            cand = f(0, c)
            if cand < ans:
                ans, how = cand, c
        return log.ret(ans, f"answer = f(0,{how})")

    ans, steps = run_call(build)
    assert ans == 13
    return _pack("Falling path II — recursion", "[[1,2,3],[4,5,6],[7,8,9]]", steps, ans)


def _fall_memo():
    def build(log):
        g, n = FALL, 3
        memo = [[None] * n for _ in range(n)]

        def f(r, c):
            if memo[r][c] is not None:
                return log.hit(f"f({r},{c})", memo[r][c])
            log.call(f"f({r},{c})")
            if r == n - 1:
                memo[r][c] = g[r][c]
                return log.base(g[r][c], f"f({r},{c}) = {show(g[r][c])}")
            best = min(f(r + 1, d) for d in range(n) if d != c)
            memo[r][c] = g[r][c] + best
            return log.ret(memo[r][c], f"f({r},{c}) = {show(memo[r][c])}")

        log.call("min over row 0")
        ans = min(f(0, c) for c in range(n))
        return log.ret(ans, f"answer = {show(ans)}")

    ans, steps = run_call(build)
    assert ans == 13
    return _pack("Falling path II — memo", "[[1,2,3],[4,5,6],[7,8,9]]", steps, ans)


def _fall_tab():
    g = FALL
    rows = [g[0][:]]
    steps = [row_step("row 0", [("dp", rows[0], 0)], "m1 = 1 at column 0, m2 = 2")]
    prev = rows[0]
    for r in range(1, 3):
        m1, m2, j = 10**18, 10**18, -1
        for d, v in enumerate(prev):
            if v < m1:
                m2, m1, j = m1, v, d
            elif v < m2:
                m2 = v
        cur = [g[r][c] + (m2 if c == j else m1) for c in range(3)]
        rows.append(cur)
        prev = cur
        steps.append(row_step(
            f"row {r}",
            [("dp", cur, cur.index(min(cur)))],
            f"m1 = {show(m1)} at column {j}, m2 = {show(m2)}",
        ))
    assert min(rows[-1]) == 13 and rows == [[1, 2, 3], [6, 6, 7], [13, 14, 15]]
    return _pack("Falling path II — table", "[[1,2,3],[4,5,6],[7,8,9]]", steps, 13)


def _fall_best():
    g = FALL
    m1, m2, j = 0, 0, -1
    steps = [chip_step("above row 0", [("m1", 0, "amber"), ("j", -1, "muted"), ("m2", 0, "amber")],
                       "a virtual row of zeros")]
    for r in range(3):
        n1, n2, nj = 10**18, 10**18, -1
        for c in range(3):
            v = g[r][c] + (m2 if c == j else m1)
            if v < n1:
                n2, n1, nj = n1, v, c
            elif v < n2:
                n2 = v
        m1, m2, j = n1, n2, nj
        steps.append(chip_step(
            f"after row {r}",
            [("m1", m1, "green"), ("j", j, "indigo"), ("m2", m2, "amber")],
            f"column {j} holds the minimum" + (" (tie: m2 = m1)" if m1 == m2 else ""),
        ))
    assert m1 == 13 and len(steps) <= 8
    return _pack("Falling path II — best and second", "[[1,2,3],[4,5,6],[7,8,9]]", steps, m1)


def _score_pair(board):
    """(score, ways) from E, score -1 if unreachable. Used by the stories."""
    n = len(board)
    sc = [[-1] * n for _ in range(n)]
    wy = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            if board[r][c] == "X":
                continue
            if r == 0 and c == 0:
                sc[r][c], wy[r][c] = 0, 1
                continue
            best, ways = -1, 0
            for pr, pc in ((r - 1, c), (r, c - 1), (r - 1, c - 1)):
                if pr < 0 or pc < 0 or sc[pr][pc] < 0:
                    continue
                s = sc[pr][pc]
                if s > best:
                    best, ways = s, wy[pr][pc]
                elif s == best:
                    ways = (ways + wy[pr][pc]) % MOD
            if best < 0:
                continue
            digit = int(board[r][c]) if board[r][c].isdigit() else 0
            sc[r][c] = best + digit
            wy[r][c] = ways
    return sc, wy


def _score_rec():
    def build(log):
        b = SCORE

        def f(r, c):
            log.call(f"f({r},{c})")
            if r < 0 or c < 0 or b[r][c] == "X":
                return log.base((-1, 0), "unreachable")
            if r == 0 and c == 0:
                return log.base((0, 1), "E scores 0")
            best, ways = -1, 0
            for dr, dc in ((-1, 0), (0, -1), (-1, -1)):
                s, w = f(r + dr, c + dc)
                if s > best:
                    best, ways = s, w
                elif s == best and s >= 0:
                    ways = (ways + w) % MOD
            if best < 0:
                return log.ret((-1, 0), f"f({r},{c}) unreachable")
            digit = int(b[r][c]) if b[r][c].isdigit() else 0
            return log.ret((best + digit, ways), f"f({r},{c}) = ({show(best + digit)}, {show(ways)})")

        return f(2, 2)

    ans, steps = run_call(build)
    assert ans == (7, 1)
    return _pack("Max score — recursion", '["E23","2X2","12S"]', steps, "7,1")


def _score_memo():
    def build(log):
        b = SCORE
        memo = {}

        def f(r, c):
            if (r, c) in memo:
                return log.hit(f"f({r},{c})", memo[(r, c)])
            log.call(f"f({r},{c})")
            if r < 0 or c < 0 or b[r][c] == "X":
                memo[(r, c)] = (-1, 0)
                return log.base((-1, 0), "unreachable")
            if r == 0 and c == 0:
                memo[(r, c)] = (0, 1)
                return log.base((0, 1), "E scores 0")
            best, ways = -1, 0
            for dr, dc in ((-1, 0), (0, -1), (-1, -1)):
                s, w = f(r + dr, c + dc)
                if s > best:
                    best, ways = s, w
                elif s == best and s >= 0:
                    ways = (ways + w) % MOD
            if best < 0:
                memo[(r, c)] = (-1, 0)
                return log.ret((-1, 0), f"f({r},{c}) unreachable")
            digit = int(b[r][c]) if b[r][c].isdigit() else 0
            memo[(r, c)] = (best + digit, ways)
            return log.ret(memo[(r, c)], f"f({r},{c}) = ({show(best + digit)}, {show(ways)})")

        return f(2, 2)

    ans, steps = run_call(build)
    assert ans == (7, 1)
    return _pack("Max score — memo", '["E23","2X2","12S"]', steps, "7,1")


def _fmt_cell(board, sc, wy, r, c, known):
    if board[r][c] == "X":
        return "X"
    if (r, c) not in known:
        return "·"
    if sc[r][c] < 0:
        return "−1"
    return f"{sc[r][c]},{wy[r][c]}"


def _score_tab():
    sc, wy = _score_pair(SCORE)
    assert (sc[2][2], wy[2][2]) == (7, 1)
    marks = {(1, 1): "red"}
    known = set()
    steps = []
    order = [(0, 0), (0, 2), (1, 2), (2, 2)]
    eqs = {
        (0, 0): "E is (0, 1); the letter scores 0",
        (0, 2): "2 + 3 along the top",
        (1, 2): "5 + 2, the obstacle is skipped",
        (2, 2): "up is (7, 1), left is (5, 1); S adds 0",
    }
    for r, c in order:
        known.add((r, c))
        # reveal the whole prefix in reading order up to this cell, so the grid is a real table
        for i in range(3):
            for j in range(3):
                if (i, j) < (r, c) or (i, j) == (r, c) or SCORE[i][j] == "X":
                    if i < r or (i == r and j <= c) or SCORE[i][j] == "X":
                        known.add((i, j))
        grid = [[_fmt_cell(SCORE, sc, wy, i, j, known) for j in range(3)] for i in range(3)]
        steps.append(grid_step(f"({r},{c})", grid, hi=(r, c), marks=marks, eq=eqs[(r, c)]))
    assert len(steps) <= 8
    return _pack("Max score — table", '["E23","2X2","12S"]', steps, "7,1")


def _score_roll():
    b = SCORE
    n = 3
    ps, pw = [-1] * n, [0] * n
    steps = []
    for r in range(n):
        cs, cw = [-1] * n, [0] * n
        for c in range(n):
            if b[r][c] == "X":
                continue
            if r == 0 and c == 0:
                cs[c], cw[c] = 0, 1
                continue
            s3 = [ps[c], cs[c - 1] if c else -1, ps[c - 1] if c else -1]
            w3 = [pw[c], cw[c - 1] if c else 0, pw[c - 1] if c else 0]
            best, ways = -1, 0
            for s, w in zip(s3, w3):
                if s < 0:
                    continue
                if s > best:
                    best, ways = s, w
                elif s == best:
                    ways = (ways + w) % MOD
            if best < 0:
                continue
            digit = int(b[r][c]) if b[r][c].isdigit() else 0
            cs[c] = best + digit
            cw[c] = ways
        steps.append(row_step(
            f"row {r}",
            [("score", [show(v) if v >= 0 else ("X" if b[r][c] == "X" else "−1") for c, v in enumerate(cs)], None),
             ("ways", [show(cw[c]) if cs[c] >= 0 else "·" for c in range(n)], None)],
            "up and up-left still sit in the previous row",
        ))
        ps, pw = cs, cw
    assert (ps[2], pw[2]) == (7, 1)
    return _pack("Max score — two rows", '["E23","2X2","12S"]', steps, "7,1")


def _paren_rec():
    def build(log):
        g = PAREN
        m, n = 4, 3

        def f(r, c, b):
            log.call(f"f({r},{c},{b})")
            b += 1 if g[r][c] == "(" else -1
            if b < 0:
                return log.base(False, "balance < 0")
            if r == m - 1 and c == n - 1:
                return log.base(b == 0, f"end balance {show(b)}")
            if r + 1 < m and f(r + 1, c, b):
                return log.ret(True, "down reaches a valid end")
            if c + 1 < n and f(r, c + 1, b):
                return log.ret(True, "right reaches a valid end")
            return log.ret(False, "both moves die")

        return f(0, 0, 0)

    ans, steps = run_call(build)
    assert ans is True
    return _pack("Parentheses path — recursion", '["(((", ")()", "(()", "(()"]', steps, "true")


def _paren_memo():
    def build(log):
        g = PAREN
        m, n = 4, 3
        memo = {}

        def f(r, c, b):
            b += 1 if g[r][c] == "(" else -1
            rem = (m - 1 - r) + (n - 1 - c)
            if b < 0 or b > rem:
                log.call(f"f({r},{c})")
                return log.base(False, "pruned")
            key = (r, c, b)
            if key in memo:
                return log.hit(f"({r},{c},{b})", memo[key])
            log.call(f"({r},{c},{b})")
            if r == m - 1 and c == n - 1:
                memo[key] = True
                return log.base(True, "balance 0 at the corner")
            ok = (r + 1 < m and f(r + 1, c, b)) or (c + 1 < n and f(r, c + 1, b))
            memo[key] = ok
            return log.ret(ok, f"({r},{c},{b}) = {str(ok).lower()}")

        return f(0, 0, 0)

    ans, steps = run_call(build)
    assert ans is True
    return _pack("Parentheses path — memo", '["(((", ")()", "(()", "(()"]', steps, "true")


def _paren_sets(g, prune):
    m, n = len(g), len(g[0])
    H = (m + n - 1) // 2
    ok = [[set() for _ in range(n)] for _ in range(m)]
    for r in range(m):
        for c in range(n):
            d = 1 if g[r][c] == "(" else -1
            rem = (m - 1 - r) + (n - 1 - c)
            limit = min(H, rem) if prune else H
            for b in range(0, limit + 1):
                pb = b - d
                if pb < 0 or pb > H:
                    continue
                if r == 0 and c == 0:
                    if pb == 0:
                        ok[r][c].add(b)
                elif (r and pb in ok[r - 1][c]) or (c and pb in ok[r][c - 1]):
                    ok[r][c].add(b)
    return ok


def _set_text(s):
    if not s:
        return "∅"
    return "{" + ",".join(str(b) for b in sorted(s)) + "}"


def _paren_tab():
    ok = _paren_sets(PAREN, prune=True)
    assert 0 in ok[3][2]
    steps = []
    for r in range(4):
        grid = [[_set_text(ok[i][c]) for c in range(3)] for i in range(r + 1)]
        # pad to a stable 4-row picture only on the last frame; earlier frames show the prefix
        steps.append(grid_step(
            f"row {r}",
            grid,
            hi=(r, 2 if r == 3 else 0),
            marks={},
            eq="keep every balance that can still be closed" if r < 3 else "the corner holds {0}",
        ))
    assert len(steps) <= 8
    return _pack("Parentheses path — balance table", '["(((", ")()", "(()", "(()"]', steps, "true")


def _paren_bits():
    g = PAREN
    m, n = 4, 3
    row = [0] * n
    steps = []
    for r in range(m):
        for c in range(n):
            incoming = 0
            if r == 0 and c == 0:
                incoming = 1
            if r:
                incoming |= row[c]
            if c:
                incoming |= row[c - 1]
            row[c] = (incoming << 1) if g[r][c] == "(" else (incoming >> 1)
        if r in (0, 1, 3):
            bits = []
            for c in range(n):
                have = [str(b) for b in range(8) if row[c] >> b & 1]
                bits.append("{" + ",".join(have) + "}" if have else "∅")
            steps.append(row_step(
                f"masks after row {r}",
                [("bits", bits, 2 if r == 3 else None)],
                "open shifts left, close shifts right",
            ))
    assert row[n - 1] & 1
    steps.append(line_step("corner bit 0", ["true"], "a close at the end drops balance 1 onto bit 0"))
    assert len(steps) <= 8
    return _pack("Parentheses path — bit masks", '["(((", ")()", "(()", "(()"]', steps, "true")


def stories():
    return {
        ("queen", "rec"): _queen_rec(),
        ("queen", "memo"): _queen_memo(),
        ("queen", "tab"): _queen_tab(),
        ("queen", "roll"): _queen_roll(),
        ("work", "rec"): _work_rec(),
        ("work", "memo"): _work_memo(),
        ("work", "tab"): _work_tab(),
        ("work", "meet"): _work_meet(),
        ("fall", "rec"): _fall_rec(),
        ("fall", "memo"): _fall_memo(),
        ("fall", "tab"): _fall_tab(),
        ("fall", "best"): _fall_best(),
        ("score", "rec"): _score_rec(),
        ("score", "memo"): _score_memo(),
        ("score", "tab"): _score_tab(),
        ("score", "roll"): _score_roll(),
        ("paren", "rec"): _paren_rec(),
        ("paren", "memo"): _paren_memo(),
        ("paren", "tab"): _paren_tab(),
        ("paren", "bits"): _paren_bits(),
    }


if __name__ == "__main__":
    data = stories()
    expect = {
        "queen": "10",
        "work": "800",
        "fall": "13",
        "score": "7,1",
        "paren": "true",
    }
    for key, (title, _inp, steps, final) in data.items():
        assert expect[key[0]] == final, (key, final)
        assert 1 <= len(steps) <= 8, (key, len(steps))
        assert all("title" in s for s in steps)
        print(key, final)
    print("keys", len(data))
