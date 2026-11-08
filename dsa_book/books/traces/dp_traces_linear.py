"""Step frames for climbing stairs, frog 1, and domino/tromino tiling.

Drawing lives elsewhere. ``stories`` returns data only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import run_call, row_step, chip_step, line_step, show

MOD = 10**9 + 7


def _seen(values, known):
    return [values[i] if i in known else "·" for i in range(len(values))]


def _mat(A):
    body = ", ".join("[" + ", ".join(show(x) for x in row) + "]" for row in A)
    return "[" + body + "]"


def _rm(R, M, e):
    return [f"R = {_mat(R)}", f"M = {_mat(M)}", f"e = {show(e)}"]


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    return (title, inp, steps, text)


def _stairs_rec():
    def build(log):
        n = 3

        def f(i):
            log.call(f"f({i})")
            if i == n:
                return log.base(1, f"f({i}) = 1")
            if i > n:
                return log.base(0, f"f({i}) = 0")
            a = f(i + 1)
            b = f(i + 2)
            return log.ret(a + b, f"f({i}) = {show(a)} + {show(b)}")

        return f(0)

    ans, steps = run_call(build)
    return _pack("Climbing stairs — recursion", "n = 3", steps, ans)


def _stairs_memo():
    def build(log):
        n = 3
        memo = [-1] * (n + 2)

        def f(i):
            if i == n:
                log.call(f"f({i})")
                return log.base(1, f"f({i}) = 1")
            if i > n:
                log.call(f"f({i})")
                return log.base(0, f"f({i}) = 0")
            if memo[i] >= 0:
                return log.hit(f"f({i})", memo[i])
            log.call(f"f({i})")
            a = f(i + 1)
            b = f(i + 2)
            memo[i] = a + b
            return log.ret(memo[i], f"f({i}) = {show(a)} + {show(b)}")

        return f(0)

    ans, steps = run_call(build)
    return _pack("Climbing stairs — memo", "n = 3", steps, ans)


def _stairs_tab():
    n = 3
    dp = [0] * (n + 1)
    dp[0] = 1
    known = {0}
    steps = [row_step("dp[0] = 1", [("dp", _seen(dp, known), 0)], "dp[0] = 1")]
    for i in range(1, n + 1):
        prev1 = dp[i - 1]
        prev2 = dp[i - 2] if i >= 2 else 0
        dp[i] = prev1 + prev2
        known.add(i)
        steps.append(row_step(
            f"dp[{i}] = {show(dp[i])}",
            [("dp", _seen(dp, known), i)],
            f"{show(dp[i])} = {show(prev1)} + {show(prev2)}",
        ))
    return _pack("Climbing stairs — tabulation", "n = 3", steps, dp[n])


def _stairs_opt1():
    prev2, prev1 = 1, 1
    steps = [chip_step(
        "dp[0], dp[1]",
        [("prev2", prev2, "amber"), ("prev1", prev1, "indigo")],
        "prev2, prev1 = 1, 1",
    )]
    for i in range(2, 4):
        nxt = prev1 + prev2
        eq = f"{show(nxt)} = {show(prev1)} + {show(prev2)}"
        prev2, prev1 = prev1, nxt
        steps.append(chip_step(
            f"i = {i}",
            [("prev2", prev2, "amber"), ("prev1", prev1, "green" if i == 3 else "indigo")],
            eq,
        ))
    steps.append(chip_step(
        "answer",
        [("prev2", prev2, "amber"), ("prev1", prev1, "green")],
        f"prev1 = {show(prev1)}",
    ))
    return _pack("Climbing stairs — optimal 1", "n = 3", steps, prev1)


def _stairs_opt2():
    def mul(A, B):
        return [
            [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]],
        ]

    R = [[1, 0], [0, 1]]
    M = [[1, 1], [1, 0]]
    e = 3
    steps = [line_step("R = I", _rm(R, M, e), "e = 3")]
    while e:
        if e & 1:
            R = mul(R, M)
            steps.append(line_step("e is odd, R = R · M", _rm(R, M, e), "R = R · M"))
        M = mul(M, M)
        e >>= 1
        steps.append(line_step("square M", _rm(R, M, e), f"e = {show(e)}"))
    steps.append(line_step(
        "dp[n] = R[0][0]",
        _rm(R, M, e),
        f"{show(R[0][0])} = R[0][0]",
    ))
    return _pack("Climbing stairs — optimal 2", "n = 3", steps, R[0][0])


def _frog_rec():
    def build(log):
        h = [10, 30, 40, 20]
        n = len(h)

        def f(i):
            log.call(f"f({i})")
            if i == n - 1:
                return log.base(0, f"f({i}) = 0")
            gap = abs(h[i + 1] - h[i])
            down = f(i + 1)
            a = down + gap
            if i + 2 < n:
                gap2 = abs(h[i + 2] - h[i])
                down2 = f(i + 2)
                b = down2 + gap2
                best = min(a, b)
                return log.ret(best, f"f({i}) = min({show(a)}, {show(b)})")
            return log.ret(a, f"f({i}) = {show(gap)} + {show(down)}")

        return f(0)

    ans, steps = run_call(build)
    return _pack("Frog 1 — recursion", "h = [10, 30, 40, 20]", steps, ans)


def _frog_memo():
    def build(log):
        h = [10, 30, 40, 20]
        n = len(h)
        memo = [-1] * n

        def f(i):
            if i == n - 1:
                log.call(f"f({i})")
                return log.base(0, f"f({i}) = 0")
            if memo[i] >= 0:
                return log.hit(f"f({i})", memo[i])
            log.call(f"f({i})")
            gap = abs(h[i + 1] - h[i])
            down = f(i + 1)
            a = down + gap
            if i + 2 < n:
                gap2 = abs(h[i + 2] - h[i])
                down2 = f(i + 2)
                b = down2 + gap2
                best = min(a, b)
                memo[i] = best
                return log.ret(best, f"f({i}) = min({show(a)}, {show(b)})")
            memo[i] = a
            return log.ret(a, f"f({i}) = {show(gap)} + {show(down)}")

        return f(0)

    ans, steps = run_call(build)
    return _pack("Frog 1 — memo", "h = [10, 30, 40, 20]", steps, ans)


def _frog_tab():
    h = [10, 30, 40, 20]
    n = len(h)
    dp = [0] * n
    known = {0}
    steps = [row_step("dp[0] = 0", [("dp", _seen(dp, known), 0)], "dp[0] = 0")]
    for i in range(1, n):
        src = dp[i - 1]
        gap = abs(h[i] - h[i - 1])
        one = src + gap
        dp[i] = one
        known.add(i)
        steps.append(row_step(
            f"dp[{i}] from {i - 1}",
            [("dp", _seen(dp, known), i)],
            f"{show(one)} = {show(src)} + {show(gap)}",
        ))
        if i >= 2:
            src2 = dp[i - 2]
            gap2 = abs(h[i] - h[i - 2])
            alt = src2 + gap2
            dp[i] = min(dp[i], alt)
            steps.append(row_step(
                f"dp[{i}] min with {i - 2}",
                [("dp", _seen(dp, known), i)],
                f"{show(dp[i])} = min({show(one)}, {show(alt)})",
            ))
    return _pack("Frog 1 — tabulation", "h = [10, 30, 40, 20]", steps, dp[-1])


def _frog_opt1():
    h = [10, 30, 40, 20]
    prev2, prev1 = 0, 0
    steps = [chip_step(
        "start",
        [("prev2", prev2, "amber"), ("prev1", prev1, "indigo")],
        "prev2, prev1 = 0, 0",
    )]
    for i in range(1, len(h)):
        one = prev1 + abs(h[i] - h[i - 1])
        cur = one
        if i >= 2:
            two = prev2 + abs(h[i] - h[i - 2])
            cur = min(cur, two)
            eq = f"{show(cur)} = min({show(one)}, {show(two)})"
        else:
            eq = f"{show(cur)} = {show(prev1)} + {show(abs(h[i] - h[i - 1]))}"
        prev2, prev1 = prev1, cur
        steps.append(chip_step(
            f"i = {i}",
            [("prev2", prev2, "amber"), ("prev1", prev1, "green" if i == len(h) - 1 else "indigo")],
            eq,
        ))
    return _pack("Frog 1 — optimal 1", "h = [10, 30, 40, 20]", steps, prev1)


def _frog_opt2():
    h = [10, 30, 40, 20]
    n = len(h)
    dp = [0] * n
    par = [-1] * n
    known = {0}
    steps = []
    for i in range(1, n):
        src = dp[i - 1]
        gap = abs(h[i] - h[i - 1])
        one = src + gap
        dp[i], par[i] = one, i - 1
        known.add(i)
        steps.append(row_step(
            f"stone {i} from {i - 1}",
            [("dp", _seen(dp, known), i), ("par", _seen(par, known), i)],
            f"{show(one)} = {show(src)} + {show(gap)}",
        ))
        if i >= 2:
            src2 = dp[i - 2]
            gap2 = abs(h[i] - h[i - 2])
            alt = src2 + gap2
            if alt < dp[i]:
                old = dp[i]
                dp[i], par[i] = alt, i - 2
                steps.append(row_step(
                    f"stone {i} from {i - 2}",
                    [("dp", _seen(dp, known), i), ("par", _seen(par, known), i)],
                    f"{show(alt)} < {show(old)}, par = {show(i - 2)}",
                ))
            else:
                steps.append(row_step(
                    f"stone {i} keeps 1-step",
                    [("dp", _seen(dp, known), i), ("par", _seen(par, known), i)],
                    f"{show(one)} ≤ {show(alt)}, par = {show(par[i])}",
                ))
    route = []
    i = n - 1
    while i >= 0:
        route.append(i)
        nxt = par[i]
        back = " → ".join(show(x) for x in route)
        lines = [
            f"dp = {' '.join(show(x) for x in dp)}",
            f"par = {' '.join(show(x) for x in par)}",
            f"back = {back}",
        ]
        if nxt < 0:
            forward = " → ".join(show(x) for x in reversed(route))
            lines.append(f"route = {forward}")
            steps.append(line_step(
                "reverse the walk",
                lines,
                f"{show(dp[-1])} · {forward}",
            ))
        else:
            steps.append(line_step(
                f"walk {show(i)}",
                lines,
                f"par[{show(i)}] = {show(nxt)}",
            ))
        i = nxt
    forward = " → ".join(show(x) for x in reversed(route))
    final = f"{show(dp[-1])} · {forward}"
    return _pack("Frog 1 — optimal 2", "h = [10, 30, 40, 20]", steps, final)


def _tiling_rec():
    def build(log):
        def g(i):
            log.call(f"g({i})")
            if i <= 0:
                return log.base(0, f"g({i}) = 0")
            if i == 1:
                return log.base(1, f"g({i}) = 1")
            a = g(i - 1)
            b = f(i - 1)
            return log.ret(a + b, f"g({i}) = {show(a)} + {show(b)}")

        def f(i):
            log.call(f"f({i})")
            if i == 0 or i == 1:
                return log.base(1, f"f({i}) = 1")
            a = f(i - 1)
            b = f(i - 2)
            c = g(i - 2)
            return log.ret(a + b + 2 * c, f"f({i}) = {show(a)} + {show(b)} + 2*{show(c)}")

        return f(3)

    ans, steps = run_call(build)
    return _pack("Tiling — recursion", "n = 3", steps, ans)


def _tiling_memo():
    def build(log):
        n = 3
        mf = [-1] * (n + 1)
        mg = [-1] * (n + 1)

        def g(i):
            if i <= 0:
                log.call(f"g({i})")
                return log.base(0, f"g({i}) = 0")
            if i == 1:
                log.call(f"g({i})")
                return log.base(1, f"g({i}) = 1")
            if mg[i] >= 0:
                return log.hit(f"g({i})", mg[i])
            log.call(f"g({i})")
            a = g(i - 1)
            b = f(i - 1)
            mg[i] = a + b
            return log.ret(mg[i], f"g({i}) = {show(a)} + {show(b)}")

        def f(i):
            if i == 0 or i == 1:
                log.call(f"f({i})")
                return log.base(1, f"f({i}) = 1")
            if mf[i] >= 0:
                return log.hit(f"f({i})", mf[i])
            log.call(f"f({i})")
            a = f(i - 1)
            b = f(i - 2)
            c = g(i - 2)
            mf[i] = a + b + 2 * c
            return log.ret(mf[i], f"f({i}) = {show(a)} + {show(b)} + 2*{show(c)}")

        return f(n)

    ans, steps = run_call(build)
    return _pack("Tiling — memo", "n = 3", steps, ans)


def _tiling_tab():
    n = 3
    f = [1, 1] + [0] * (n - 1)
    g = [0, 1] + [0] * (n - 1)
    known_f = {0, 1}
    known_g = {0, 1}
    steps = [row_step(
        "bases",
        [("f", _seen(f, known_f), None), ("g", _seen(g, known_g), None)],
        "f[0] = 1, f[1] = 1, g[0] = 0, g[1] = 1",
    )]
    for i in range(2, n + 1):
        f[i] = (f[i - 1] + f[i - 2] + 2 * g[i - 2]) % MOD
        known_f.add(i)
        steps.append(row_step(
            f"f[{i}]",
            [("f", _seen(f, known_f), i), ("g", _seen(g, known_g), None)],
            f"{show(f[i])} = ({show(f[i - 1])} + {show(f[i - 2])} + 2*{show(g[i - 2])}) mod {MOD}",
        ))
        g[i] = (g[i - 1] + f[i - 1]) % MOD
        known_g.add(i)
        steps.append(row_step(
            f"g[{i}]",
            [("f", _seen(f, known_f), None), ("g", _seen(g, known_g), i)],
            f"{show(g[i])} = ({show(g[i - 1])} + {show(f[i - 1])}) mod {MOD}",
        ))
    return _pack("Tiling — tabulation", "n = 3", steps, f[n])


def _tiling_opt1():
    a, b, c = 1, 1, 2
    steps = [chip_step(
        "f[0], f[1], f[2]",
        [("a", a, "amber"), ("b", b, "indigo"), ("c", c, "indigo")],
        "a, b, c = 1, 1, 2",
    )]
    nxt = (2 * c + a) % MOD
    steps.append(chip_step(
        "next = 2*c + a",
        [("a", a, "amber"), ("b", b, "indigo"), ("c", c, "indigo"), ("next", nxt, "green")],
        f"{show(nxt)} = (2*{show(c)} + {show(a)}) mod {MOD}",
    ))
    a, b, c = b, c, nxt
    steps.append(chip_step(
        "a, b, c = b, c, next",
        [("a", a, "amber"), ("b", b, "indigo"), ("c", c, "green")],
        f"a, b, c = {show(a)}, {show(b)}, {show(c)}",
    ))
    steps.append(chip_step(
        "answer",
        [("c", c, "green")],
        f"c = {show(c)}",
    ))
    return _pack("Tiling — optimal 1", "n = 3", steps, c)


def _tiling_opt2():
    def mul(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(3)) % MOD for j in range(3)] for i in range(3)]

    n = 3
    M = [[2, 0, 1], [1, 0, 0], [0, 1, 0]]
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    e = max(n - 2, 0)
    steps = [line_step("R = I", _rm(R, M, e), "e = max(n - 2, 0)")]
    while e:
        if e & 1:
            R = mul(R, M)
            steps.append(line_step("e is odd, R = R · M", _rm(R, M, e), "R = R · M"))
        M = mul(M, M)
        e >>= 1
        steps.append(line_step("square M", _rm(R, M, e), f"e = {show(e)}"))
    ans = (R[0][0] * 2 + R[0][1] * 1 + R[0][2] * 1) % MOD if n >= 2 else 1
    steps.append(line_step(
        "f[n] from row 0",
        _rm(R, M, e) + ["f[n] = R[0] · (f[2], f[1], f[0])"],
        f"{show(ans)} = ({show(R[0][0])}·2 + {show(R[0][1])}·1 + {show(R[0][2])}·1) mod {MOD}",
    ))
    return _pack("Tiling — optimal 2", "n = 3", steps, ans)


def stories():
    return {
        ("stairs", "rec"): _stairs_rec(),
        ("stairs", "memo"): _stairs_memo(),
        ("stairs", "tab"): _stairs_tab(),
        ("stairs", "opt1"): _stairs_opt1(),
        ("stairs", "opt2"): _stairs_opt2(),
        ("frog", "rec"): _frog_rec(),
        ("frog", "memo"): _frog_memo(),
        ("frog", "tab"): _frog_tab(),
        ("frog", "opt1"): _frog_opt1(),
        ("frog", "opt2"): _frog_opt2(),
        ("tiling", "rec"): _tiling_rec(),
        ("tiling", "memo"): _tiling_memo(),
        ("tiling", "tab"): _tiling_tab(),
        ("tiling", "opt1"): _tiling_opt1(),
        ("tiling", "opt2"): _tiling_opt2(),
    }


if __name__ == "__main__":
    data = stories()
    forms = ("rec", "memo", "tab", "opt1", "opt2")
    for form in forms:
        assert data[("stairs", form)][3] == "3"
        assert data[("tiling", form)][3] == "5"
    for form in ("rec", "memo", "tab", "opt1"):
        assert data[("frog", form)][3] == "30"
    assert data[("frog", "opt2")][3].startswith("30")
    for key, (_title, _inp, steps, _final) in data.items():
        assert 3 <= len(steps) <= 8, (key, len(steps))
    assert any("memo" in step["title"] for step in data[("stairs", "memo")][2])
    for key, (_title, _inp, _steps, final) in data.items():
        print(key, final)
