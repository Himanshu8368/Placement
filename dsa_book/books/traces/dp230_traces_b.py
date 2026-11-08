"""Step boards for the back half of the one-dimensional DP chapter.

Patterns 3–5: counting under rules, reachability, suffix and partition.
``stories`` returns ``(problem, form) -> (title, input line, steps, final)``.
Drawing lives in ``dsa_viz.story``. This module only records frames.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import chip_step, grid_step, line_step, row_step, run_call, show

MOD = 10**9 + 7
INF = 10**9

KNIGHT_MOVES = (
    (4, 6), (6, 8), (7, 9), (4, 8), (0, 3, 9),
    (), (0, 1, 7), (2, 6), (1, 3), (2, 4),
)


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    steps = list(steps)
    if len(steps) > 8:
        steps = steps[:7] + steps[-1:]
    return (title, inp, steps, text)


def _seen(values, known):
    return [values[i] if i in known else "·" for i in range(len(values))]


def one_digit(c):
    if c == "*":
        return 9
    return 0 if c == "0" else 1


def two_digit(a, b):
    if a == "*" and b == "*":
        return 15
    if a == "*":
        return 2 if b <= "6" else 1
    if b == "*":
        return 9 if a == "1" else (6 if a == "2" else 0)
    value = (ord(a) - 48) * 10 + (ord(b) - 48)
    return 1 if 10 <= value <= 26 else 0


def _decode_rec():
    def build(log):
        s = "*"

        def f(i):
            log.call(f"f({i})")
            if i == len(s):
                return log.base(1, "f(1) = 1")
            res = one_digit(s[i]) * f(i + 1) % MOD
            if i + 1 < len(s):
                res = (res + two_digit(s[i], s[i + 1]) * f(i + 2)) % MOD
            return log.ret(res, f"f({i}) = {show(res)}")

        return f(0)

    ans, steps = run_call(build)
    return _pack('Decode ways II — recursion', 's = "*"', steps, ans)


def _decode_memo():
    def build(log):
        s = "1*"
        memo = [-1] * (len(s) + 1)

        def f(i):
            if i == len(s):
                log.call(f"f({i})")
                return log.base(1, "f(2) = 1")
            if memo[i] >= 0:
                return log.hit(f"f({i})", memo[i])
            log.call(f"f({i})")
            res = one_digit(s[i]) * f(i + 1) % MOD
            if i + 1 < len(s):
                res = (res + two_digit(s[i], s[i + 1]) * f(i + 2)) % MOD
            memo[i] = res
            return log.ret(res, f"f({i}) = {show(res)}")

        return f(0)

    ans, steps = run_call(build)
    return _pack('Decode ways II — memo', 's = "1*"', steps, ans)


def _decode_tab():
    s = "2*"
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    known = {0}
    steps = [row_step("empty prefix", [("dp", _seen(dp, known), 0)], "dp[0] = 1")]
    for i in range(1, n + 1):
        single = one_digit(s[i - 1])
        pair = two_digit(s[i - 2], s[i - 1]) if i >= 2 else 0
        dp[i] = (single * dp[i - 1] + (pair * dp[i - 2] if i >= 2 else 0)) % MOD
        known.add(i)
        eq = f"{show(dp[i])} = {show(single)}·dp[{i - 1}]"
        if i >= 2:
            eq += f" + {show(pair)}·dp[{i - 2}]"
        steps.append(row_step(f"i = {i}", [("dp", _seen(dp, known), i)], eq))
    return _pack('Decode ways II — table', 's = "2*"', steps, dp[n])


def _decode_roll():
    s = "1*"
    prev2, prev1 = 1, one_digit(s[0])
    steps = [chip_step(
        "dp[0], dp[1]",
        [("prev2", prev2, "amber"), ("prev1", prev1, "indigo")],
        "prev2 = 1, prev1 = one(s[0])",
    )]
    for i in range(1, len(s)):
        cur = (one_digit(s[i]) * prev1 + two_digit(s[i - 1], s[i]) * prev2) % MOD
        eq = f"{show(cur)} = {show(one_digit(s[i]))}·{show(prev1)} + {show(two_digit(s[i - 1], s[i]))}·{show(prev2)}"
        prev2, prev1 = prev1, cur
        steps.append(chip_step(
            f"read s[{i}] = {s[i]}",
            [("prev2", prev2, "amber"), ("prev1", prev1, "green")],
            eq,
        ))
    return _pack('Decode ways II — rolling', 's = "1*"', steps, prev1)


def _knight_rec():
    def build(log):
        n = 2

        def f(d, length):
            log.call(f"{d},{length}")
            if length == 1:
                return log.base(1, f"{d},1 = 1")
            res = 0
            for e in KNIGHT_MOVES[d]:
                res = (res + f(e, length - 1)) % MOD
            return log.ret(res, f"{d},{length} = {show(res)}")

        log.call("sum")
        total = 0
        for d in range(10):
            total = (total + f(d, n)) % MOD
        return log.ret(total, f"sum = {show(total)}")

    ans, steps = run_call(build)
    return _pack("Knight dialer — recursion", "n = 2", steps, ans)


def _knight_memo():
    def build(log):
        n = 2
        memo = [[-1] * (n + 1) for _ in range(10)]

        def f(d, length):
            if length == 1:
                if memo[d][1] >= 0:
                    return log.hit(f"{d},1", memo[d][1])
                log.call(f"{d},1")
                memo[d][1] = 1
                return log.base(1, f"{d},1 = 1")
            if memo[d][length] >= 0:
                return log.hit(f"{d},{length}", memo[d][length])
            log.call(f"{d},{length}")
            res = 0
            for e in KNIGHT_MOVES[d]:
                res = (res + f(e, length - 1)) % MOD
            memo[d][length] = res
            return log.ret(res, f"{d},{length} = {show(res)}")

        log.call("sum")
        total = 0
        for d in range(10):
            total = (total + f(d, n)) % MOD
        return log.ret(total, f"sum = {show(total)}")

    ans, steps = run_call(build)
    return _pack("Knight dialer — memo", "n = 2", steps, ans)


def _knight_tab():
    n = 2
    cur = [1] * 10
    steps = [row_step("length 1", [("dp", [show(v) for v in cur], None)], "each key once, sum 10")]
    nxt = [0] * 10
    for d in range(10):
        for e in KNIGHT_MOVES[d]:
            nxt[e] = (nxt[e] + cur[d]) % MOD
    steps.append(row_step(
        "length 2",
        [("dp", [show(v) for v in nxt], 4)],
        f"sum = {show(sum(nxt) % MOD)}",
    ))
    return _pack("Knight dialer — table", "n = 2", steps, sum(nxt) % MOD)


def _knight_sym():
    a = b = c = d = 1
    steps = [chip_step(
        "length 1",
        [("a", a, "green"), ("b", b, "indigo"), ("c", c, "amber"), ("d", d, "violet")],
        "corners, {2,8}, {4,6}, key 0; plus key 5 → 10",
    )]
    na = (b + c) % MOD
    nb = 2 * a % MOD
    nc = (2 * a + d) % MOD
    nd = 2 * c % MOD
    a, b, c, d = na, nb, nc, nd
    total = (4 * a + 2 * b + 2 * c + d) % MOD
    steps.append(chip_step(
        "length 2",
        [("a", a, "green"), ("b", b, "indigo"), ("c", c, "amber"), ("d", d, "violet")],
        f"4a+2b+2c+d = {show(total)}",
    ))
    return _pack("Knight dialer — symmetry", "n = 2", steps, total)


def _jump_rec():
    def build(log):
        a = [2, 3, 1, 1, 4]
        n = len(a)

        def f(i):
            log.call(f"f({i})")
            if i == n - 1:
                return log.base(0, "f(4) = 0")
            best = INF
            for j in range(i + 1, min(n, i + a[i] + 1)):
                best = min(best, 1 + f(j))
            return log.ret(best, f"f({i}) = {show(best)}")

        return f(0)

    ans, steps = run_call(build)
    return _pack("Jump game II — recursion", "nums = [2, 3, 1, 1, 4]", steps, ans)


def _jump_memo():
    def build(log):
        a = [2, 3, 1, 1, 4]
        n = len(a)
        memo = [-1] * n

        def f(i):
            if i == n - 1:
                log.call("f(4)")
                return log.base(0, "f(4) = 0")
            if memo[i] >= 0:
                return log.hit(f"f({i})", memo[i])
            log.call(f"f({i})")
            best = INF
            for j in range(i + 1, min(n, i + a[i] + 1)):
                best = min(best, 1 + f(j))
            memo[i] = best
            return log.ret(best, f"f({i}) = {show(best)}")

        return f(0)

    ans, steps = run_call(build)
    return _pack("Jump game II — memo", "nums = [2, 3, 1, 1, 4]", steps, ans)


def _jump_tab():
    a = [2, 3, 1, 1, 4]
    n = len(a)
    dp = [INF] * n
    dp[0] = 0
    known = {0}
    steps = [row_step("start", [("dp", _seen(dp, known), 0)], "dp[0] = 0")]
    for i in range(n):
        if dp[i] >= INF:
            continue
        changed = False
        for j in range(i + 1, min(n, i + a[i] + 1)):
            if dp[i] + 1 < dp[j]:
                dp[j] = dp[i] + 1
                known.add(j)
                changed = True
        if changed:
            steps.append(row_step(
                f"push from {i}",
                [("dp", _seen(dp, known), None)],
                f"dp[{i}] + 1 = {show(dp[i] + 1)}",
            ))
    return _pack("Jump game II — table", "nums = [2, 3, 1, 1, 4]", steps, dp[-1])


def _jump_bfs():
    a = [2, 3, 1, 1, 4]
    n = len(a)
    jumps = layer_end = farthest = 0
    steps = [chip_step(
        "before the scan",
        [("jumps", jumps, "amber"), ("layerEnd", layer_end, "indigo"), ("farthest", farthest, "violet")],
        "layer 0 is just index 0",
    )]
    for i in range(n - 1):
        farthest = max(farthest, i + a[i])
        closed = i == layer_end
        if closed:
            jumps += 1
            layer_end = farthest
        steps.append(chip_step(
            f"i = {i}" + ("  close layer" if closed else ""),
            [("jumps", jumps, "green" if closed else "amber"),
             ("layerEnd", layer_end, "indigo"),
             ("farthest", farthest, "violet")],
            f"farthest = max(farthest, {i}+{a[i]})",
        ))
    return _pack("Jump game II — layers", "nums = [2, 3, 1, 1, 4]", steps, jumps)


def _frog_search(stones, log=None):
    pos = {p: i for i, p in enumerate(stones)}

    def can(i, k):
        label = f"({stones[i]},{k})"
        if log is not None:
            log.call(label)
        if i == len(stones) - 1:
            return log.base(1, f"{label} = yes") if log is not None else True
        for s in (k - 1, k, k + 1):
            if s <= 0:
                continue
            j = pos.get(stones[i] + s)
            if j is not None and can(j, s):
                if log is not None:
                    return log.ret(1, f"{label} = yes")
                return True
        if log is not None:
            return log.ret(0, f"{label} = no")
        return False

    return can(0, 0)


def _frog_rec():
    stones = [0, 1, 3, 5, 6, 8, 12, 17]

    def build(log):
        return _frog_search(stones, log)

    ans, steps = run_call(build)
    return _pack("Frog jump — recursion", "stones = [0, 1, 3, 5, 6, 8, 12, 17]", steps, "true" if ans else "false")


def _frog_memo():
    stones = [0, 1, 2, 3, 4, 8, 9, 11]

    def build(log):
        pos = {p: i for i, p in enumerate(stones)}
        n = len(stones)
        memo = [[-1] * (n + 1) for _ in range(n)]

        def can(i, k):
            label = f"({stones[i]},{k})"
            if i == n - 1:
                log.call(label)
                return log.base(1, f"{label} = yes")
            if memo[i][k] >= 0:
                return log.hit(label, memo[i][k])
            log.call(label)
            memo[i][k] = 0
            for s in (k - 1, k, k + 1):
                if s <= 0:
                    continue
                j = pos.get(stones[i] + s)
                if j is not None and can(j, s):
                    memo[i][k] = 1
                    return log.ret(1, f"{label} = yes")
            return log.ret(0, f"{label} = no")

        return can(0, 0)

    ans, steps = run_call(build)
    return _pack("Frog jump — memo", "stones = [0, 1, 2, 3, 4, 8, 9, 11]", steps, "true" if ans else "false")


def _frog_push():
    stones = [0, 1, 3, 5, 6, 8, 12, 17]
    jumps = {p: set() for p in stones}
    jumps[0].add(0)
    steps = []
    for p in stones:
        for k in list(jumps[p]):
            for s in (k - 1, k, k + 1):
                if s > 0 and (p + s) in jumps:
                    jumps[p + s].add(s)
        shown = "  ".join(f"{q}:{sorted(jumps[q]) or '·'}" for q in stones)
        steps.append(line_step(f"after {p}", [shown], "push k-1, k, k+1 onto a stone"))
    answer = "true" if jumps[stones[-1]] else "false"
    return _pack("Frog jump — push set", "stones = [0, 1, 3, 5, 6, 8, 12, 17]", steps, answer)


def _frog_table():
    stones = [0, 1, 3, 5, 6, 8, 12, 17]
    n = len(stones)
    dp = [[0] * (n + 1) for _ in range(n)]
    dp[0][0] = 1
    steps = [line_step("start", ["(stone 0, k 0)"], "first jump will be 1")]
    for j in range(1, n):
        for i in range(j - 1, -1, -1):
            k = stones[j] - stones[i]
            if k > i + 1:
                break
            if dp[i][k - 1] or dp[i][k] or dp[i][k + 1]:
                dp[j][k] = 1
        reached = [show(k) for k in range(n + 1) if dp[j][k]]
        steps.append(line_step(
            f"stone {stones[j]}",
            ["k in {" + ", ".join(reached) + "}" if reached else "none"],
            f"jump = stones[{j}] - stones[i], stop when k > i+1",
        ))
    answer = "true" if any(dp[n - 1]) else "false"
    return _pack("Frog jump — index table", "stones = [0, 1, 3, 5, 6, 8, 12, 17]", steps, answer)


def _brain_walk(questions, log=None, memo=None):
    n = len(questions)

    def f(i):
        if log is not None:
            if memo is not None and i < n and memo[i] >= 0:
                return log.hit(f"f({i})", memo[i])
            log.call(f"f({i})" if i < n else "f(n)")
        if i >= n:
            return log.base(0, "f(n) = 0") if log is not None else 0
        skip = f(i + 1)
        take = questions[i][0] + f(i + questions[i][1] + 1)
        best = max(skip, take)
        if memo is not None:
            memo[i] = best
        if log is not None:
            return log.ret(best, f"f({i}) = {show(best)}")
        return best

    return f(0)


def _brain_rec():
    q = [[3, 2], [4, 3], [4, 4], [2, 5]]

    def build(log):
        return _brain_walk(q, log)

    ans, steps = run_call(build)
    return _pack("Brainpower — recursion", "[[3, 2], [4, 3], [4, 4], [2, 5]]", steps, ans)


def _brain_memo():
    q = [[3, 2], [4, 3], [4, 4], [2, 5]]

    def build(log):
        return _brain_walk(q, log, memo=[-1] * len(q))

    ans, steps = run_call(build)
    return _pack("Brainpower — memo", "[[3, 2], [4, 3], [4, 4], [2, 5]]", steps, ans)


def _brain_tab():
    q = [[3, 2], [4, 3], [4, 4], [2, 5]]
    n = len(q)
    f = [0] * (n + 1)
    known = {n}
    steps = [row_step("f[n] = 0", [("f", _seen(f, known), n)], "nothing left")]
    for i in range(n - 1, -1, -1):
        nxt = min(n, i + q[i][1] + 1)
        f[i] = max(f[i + 1], q[i][0] + f[nxt])
        known.add(i)
        steps.append(row_step(
            f"f[{i}]",
            [("f", _seen(f, known), i)],
            f"max(f[{i + 1}]={show(f[i + 1])}, {q[i][0]}+f[{nxt}]={show(q[i][0] + f[nxt])})",
        ))
    return _pack("Brainpower — suffix table", "[[3, 2], [4, 3], [4, 4], [2, 5]]", steps, f[0])


def _brain_push():
    q = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    n = len(q)
    g = [0] * (n + 1)
    best = 0
    steps = []
    for i in range(n):
        best = max(best, g[i])
        nxt = min(n, i + q[i][1] + 1)
        g[nxt] = max(g[nxt], best + q[i][0])
        steps.append(line_step(
            f"question {i}",
            [f"best = {show(best)}", f"push {show(best + q[i][0])} → g[{nxt}] = {show(g[nxt])}"],
            "skip arrows collapse into best",
        ))
    answer = max(best, g[n])
    steps.append(line_step("answer", [f"max(best, g[{n}]) = {show(answer)}"], "sample 2"))
    return _pack("Brainpower — push", "[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]", steps, answer)


def _shelf_search(books, width, log=None, memo=None):
    def solve(i):
        if log is not None:
            if memo is not None and memo[i] >= 0:
                return log.hit(f"s({i})", memo[i])
            log.call(f"s({i})")
        if i == 0:
            return log.base(0, "s(0) = 0") if log is not None else 0
        best = INF
        thick = height = 0
        for j in range(i - 1, -1, -1):
            thick += books[j][0]
            if thick > width:
                break
            height = max(height, books[j][1])
            best = min(best, solve(j) + height)
        if memo is not None:
            memo[i] = best
        if log is not None:
            return log.ret(best, f"s({i}) = {show(best)}")
        return best

    return solve(len(books))


def _shelf_rec():
    books = [[1, 1], [2, 3], [2, 3], [1, 1], [1, 1], [1, 1], [1, 2]]

    def build(log):
        return _shelf_search(books, 4, log)

    ans, steps = run_call(build)
    return _pack("Bookcase shelves — recursion", "width = 4, 7 books", steps, ans)


def _shelf_memo():
    books = [[1, 3], [2, 4], [3, 2]]

    def build(log):
        return _shelf_search(books, 6, log, memo=[-1] * (len(books) + 1))

    ans, steps = run_call(build)
    return _pack("Bookcase shelves — memo", "[[1, 3], [2, 4], [3, 2]], width 6", steps, ans)


def _shelf_fill(books, width):
    n = len(books)
    dp = [INF] * (n + 1)
    dp[0] = 0
    steps = [row_step("no books", [("dp", ["0"] + ["·"] * n, 0)], "dp[0] = 0")]
    for i in range(1, n + 1):
        thick = height = 0
        for j in range(i - 1, -1, -1):
            thick += books[j][0]
            if thick > width:
                break
            height = max(height, books[j][1])
            dp[i] = min(dp[i], dp[j] + height)
        shown = [show(dp[t]) if dp[t] < INF else "·" for t in range(n + 1)]
        steps.append(row_step(f"dp[{i}] = {show(dp[i])}", [("dp", shown, i)], "last shelf grows left"))
    return dp, steps


def _shelf_tab():
    books = [[1, 1], [2, 3], [2, 3], [1, 1], [1, 1], [1, 1], [1, 2]]
    dp, steps = _shelf_fill(books, 4)
    return _pack("Bookcase shelves — table", "width = 4", steps, dp[-1])


def _shelf_roll():
    books = [[1, 1], [2, 3], [2, 3], [1, 1], [1, 1], [1, 1], [1, 2]]
    width = 4
    n = len(books)
    ring_n = min(n, width) + 1
    ring = [0] * ring_n
    steps = [chip_step("window", [("R", ring_n, "amber"), ("dp[0]", 0, "indigo")], "keep min(n, W)+1 cells")]
    for i in range(1, n + 1):
        best = INF
        thick = height = 0
        for j in range(i - 1, -1, -1):
            thick += books[j][0]
            if thick > width:
                break
            height = max(height, books[j][1])
            best = min(best, ring[j % ring_n] + height)
        ring[i % ring_n] = best
        if i in (1, 3, n):
            steps.append(chip_step(
                f"i = {i}",
                [("dp[i]", best, "green" if i == n else "indigo"), ("slot", i % ring_n, "violet")],
                f"ring[{i} mod {ring_n}] = {show(best)}",
            ))
    return _pack("Bookcase shelves — rolling", "width = 4", steps, ring[n % ring_n])


def _diff_search(a, d, log=None, memo=None):
    def solve(k, i):
        label = f"s({k},{i})"
        if log is not None:
            if memo is not None and memo[k][i] >= 0:
                return log.hit(label, memo[k][i])
            log.call(label)
        if k == 0:
            val = 0 if i == 0 else INF
            if log is not None:
                return log.base(val, f"{label} = {show(val)}")
            return val
        if i < k:
            if memo is not None:
                memo[k][i] = INF
            if log is not None:
                return log.ret(INF, f"{label} impossible")
            return INF
        best = INF
        mx = 0
        for j in range(i - 1, k - 2, -1):
            mx = max(mx, a[j])
            before = solve(k - 1, j)
            if before < INF:
                best = min(best, before + mx)
        if memo is not None:
            memo[k][i] = best
        if log is not None:
            shown = best if best < INF else "∞"
            return log.ret(best if best < INF else INF, f"{label} = {shown}")
        return best

    return solve(d, len(a))


def _diff_rec():
    a = [6, 5, 4, 3, 2, 1]

    def build(log):
        return _diff_search(a, 2, log)

    ans, steps = run_call(build)
    return _pack("Job schedule — recursion", "[6, 5, 4, 3, 2, 1], d = 2", steps, ans)


def _diff_memo():
    a = [1, 1, 1]

    def build(log):
        memo = [[-1] * (len(a) + 1) for _ in range(4)]
        return _diff_search(a, 3, log, memo)

    ans, steps = run_call(build)
    return _pack("Job schedule — memo", "[1, 1, 1], d = 3", steps, ans)


def _diff_layers(a, d):
    n = len(a)
    dp = [[INF] * (n + 1) for _ in range(d + 1)]
    dp[0][0] = 0
    grids = []
    for k in range(1, d + 1):
        for i in range(k, n + 1):
            mx = 0
            for j in range(i - 1, k - 2, -1):
                mx = max(mx, a[j])
                if dp[k - 1][j] < INF:
                    dp[k][i] = min(dp[k][i], dp[k - 1][j] + mx)
        grid = [[show(dp[r][c]) if dp[r][c] < INF else "∞" for c in range(n + 1)] for r in range(k + 1)]
        grids.append(grid_step(f"after day-count {k}", grid, hi=(k, n), eq=f"dp[{k}][{n}]"))
    return dp, grids


def _diff_tab():
    a = [6, 5, 4, 3, 2, 1]
    dp, steps = _diff_layers(a, 2)
    return _pack("Job schedule — layers", "[6, 5, 4, 3, 2, 1], d = 2", steps, dp[2][len(a)])


def _diff_roll():
    a = [6, 5, 4, 3, 2, 1]
    d = 2
    n = len(a)
    f = [INF] * (n + 1)
    f[0] = 0
    steps = [row_step("layer 0", [("f", ["0"] + ["∞"] * n, 0)], "0 days, 0 jobs")]
    for k in range(1, d + 1):
        for i in range(n, k - 1, -1):
            best = INF
            mx = 0
            for j in range(i - 1, k - 2, -1):
                mx = max(mx, a[j])
                if f[j] < INF:
                    best = min(best, f[j] + mx)
            f[i] = best
        f[k - 1] = INF
        shown = [show(v) if v < INF else "∞" for v in f]
        steps.append(row_step(f"layer {k}", [("f", shown, n)], "i runs downward, so f[j < i] is still the previous layer"))
    return _pack("Job schedule — rolling layer", "[6, 5, 4, 3, 2, 1], d = 2", steps, f[n])


def stories():
    """Every solved form in the back half of chapter 230."""
    return {
        ("decode", "rec"): _decode_rec(),
        ("decode", "memo"): _decode_memo(),
        ("decode", "tab"): _decode_tab(),
        ("decode", "roll"): _decode_roll(),
        ("knight", "rec"): _knight_rec(),
        ("knight", "memo"): _knight_memo(),
        ("knight", "tab"): _knight_tab(),
        ("knight", "sym"): _knight_sym(),
        ("jump2", "rec"): _jump_rec(),
        ("jump2", "memo"): _jump_memo(),
        ("jump2", "tab"): _jump_tab(),
        ("jump2", "bfs"): _jump_bfs(),
        ("frog", "rec"): _frog_rec(),
        ("frog", "memo"): _frog_memo(),
        ("frog", "push"): _frog_push(),
        ("frog", "table"): _frog_table(),
        ("brain", "rec"): _brain_rec(),
        ("brain", "memo"): _brain_memo(),
        ("brain", "tab"): _brain_tab(),
        ("brain", "push"): _brain_push(),
        ("shelf", "rec"): _shelf_rec(),
        ("shelf", "memo"): _shelf_memo(),
        ("shelf", "tab"): _shelf_tab(),
        ("shelf", "roll"): _shelf_roll(),
        ("diff", "rec"): _diff_rec(),
        ("diff", "memo"): _diff_memo(),
        ("diff", "tab"): _diff_tab(),
        ("diff", "roll"): _diff_roll(),
    }


if __name__ == "__main__":
    data = stories()
    for key in data:
        title, inp, steps, final = data[key]
        assert 1 <= len(steps) <= 8, (key, len(steps))
        print(f"{key[0]:7} {key[1]:6} steps={len(steps):2} final={final!r}  {inp}")
