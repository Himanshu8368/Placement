"""Step boards for house robber and boredom. Records frames only; no drawing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import run_call, row_step, chip_step, show

ROBBER = [2, 7, 9]
BOREDOM = [1, 2, 2]


def _pts(a):
    """Value v scores v * (how many copies)."""
    mx = max(a)
    pts = [0] * (mx + 1)
    for x in a:
        pts[x] += x
    return pts


def _pair(prev2, prev1):
    return [("prev2", prev2, "indigo"), ("prev1", prev1, "green")]


def _robber_rec():
    a = list(ROBBER)

    def build(log):
        def f(i):
            log.call(f"f({i})")
            if i < 0:
                return log.base(0)
            skip = f(i - 1)
            back = f(i - 2)
            best = max(skip, a[i] + back)
            return log.ret(best, f"f({i}) = max({skip}, {a[i]}+{back})")

        return f(len(a) - 1)

    answer, steps = run_call(build)
    return "House robber — recursion", f"a = {a}", steps, show(answer)


def _robber_memo():
    a = list(ROBBER)

    def build(log):
        memo = [-1] * (len(a) + 1)

        def f(i):
            # i < 0 returns before anything is stored, so it is a base every time.
            if i < 0:
                log.call(f"f({i})")
                return log.base(0)
            if memo[i] >= 0:
                return log.hit(f"f({i})", memo[i])
            log.call(f"f({i})")
            skip = f(i - 1)
            back = f(i - 2)
            best = max(skip, a[i] + back)
            memo[i] = best
            return log.ret(best, f"f({i}) = max({skip}, {a[i]}+{back})")

        return f(len(a) - 1)

    answer, steps = run_call(build)
    return "House robber — memo", f"a = {a}", steps, show(answer)


def _robber_tab():
    a = list(ROBBER)
    dp = [0] * (len(a) + 1)
    padded = ["·", *a]
    steps = [
        row_step(
            "dp[0] = 0",
            [("a", padded, None), ("dp", dp[:], 0)],
            eq="dp[0] = 0",
        )
    ]
    for i, x in enumerate(a, 1):
        earlier = dp[i - 2] if i >= 2 else 0
        dp[i] = max(dp[i - 1], x + earlier)
        steps.append(
            row_step(
                f"dp[{i}] = {dp[i]}",
                [("a", padded, i), ("dp", dp[:], i)],
                eq=f"{dp[i]} = max({dp[i - 1]}, {x}+{earlier})",
            )
        )
    return "House robber — tabulation", f"a = {a}", steps, show(dp[-1])


def _robber_opt1():
    a = list(ROBBER)
    prev2, prev1 = 0, 0
    steps = [
        chip_step(
            "prev2, prev1 = 0, 0",
            _pair(prev2, prev1),
            eq="prev2, prev1 = 0, 0",
        )
    ]
    for i, x in enumerate(a):
        old2, old1 = prev2, prev1
        prev2, prev1 = old1, max(old1, x + old2)
        steps.append(
            chip_step(
                f"a[{i}] = {x}",
                _pair(prev2, prev1),
                eq=f"{prev1} = max({old1}, {x}+{old2})",
            )
        )
    return "House robber — optimal", f"a = {a}", steps, show(prev1)


def _robber_opt2():
    a = list(ROBBER)
    dp = [0] * (len(a) + 1)
    for i, x in enumerate(a, 1):
        earlier = dp[i - 2] if i >= 2 else 0
        dp[i] = max(dp[i - 1], x + earlier)
    padded = ["·", *a]
    taken = ["·"] * len(dp)
    chosen = []
    i = len(a)
    shown = ", ".join(show(c) for c in dp)
    steps = [
        row_step(
            f"dp = [{shown}]",
            [("a", padded, None), ("dp", dp[:], i)],
            eq=f"start at i = {i}",
        )
    ]
    while i > 0:
        if dp[i] == dp[i - 1]:
            steps.append(
                row_step(
                    f"Skip house {i - 1}",
                    [("a", padded, i), ("dp", dp[:], i), ("taken", taken[:], None)],
                    eq=f"{dp[i]} = {dp[i - 1]} → i = {i - 1}",
                )
            )
            i -= 1
        else:
            house = i - 1
            chosen.append(house)
            taken[i] = "take"
            steps.append(
                row_step(
                    f"Take house {house}",
                    [("a", padded, i), ("dp", dp[:], i), ("taken", taken[:], i)],
                    eq=f"{dp[i]} ≠ {dp[i - 1]} → house {house}",
                )
            )
            i -= 2
    houses = chosen[::-1]
    house_list = ", ".join(str(h) for h in houses)
    final = f"{show(dp[-1])} · houses {house_list}"
    steps.append(
        row_step(
            f"{show(dp[-1])} = houses {house_list}",
            [("a", padded, None), ("dp", dp[:], None), ("taken", taken[:], None)],
            eq=f"houses {house_list}",
        )
    )
    return "House robber — houses", f"a = {a}", steps, final


def _boredom_rec():
    a = list(BOREDOM)
    pts = _pts(a)

    def build(log):
        def f(v):
            log.call(f"f({v})")
            if v <= 0:
                return log.base(0)
            skip = f(v - 1)
            back = f(v - 2)
            best = max(skip, pts[v] + back)
            return log.ret(best, f"f({v}) = max({skip}, {pts[v]}+{back})")

        return f(len(pts) - 1)

    answer, steps = run_call(build)
    return "Boredom — recursion", f"a = {a}", steps, show(answer)


def _boredom_memo():
    a = list(BOREDOM)
    pts = _pts(a)

    def build(log):
        memo = [-1] * len(pts)

        def f(v):
            # v <= 0 returns before memo is written, so it is never a hit.
            if v <= 0:
                log.call(f"f({v})")
                return log.base(0)
            if memo[v] >= 0:
                return log.hit(f"f({v})", memo[v])
            log.call(f"f({v})")
            skip = f(v - 1)
            back = f(v - 2)
            best = max(skip, pts[v] + back)
            memo[v] = best
            return log.ret(best, f"f({v}) = max({skip}, {pts[v]}+{back})")

        return f(len(pts) - 1)

    answer, steps = run_call(build)
    return "Boredom — memo", f"a = {a}", steps, show(answer)


def _boredom_tab():
    a = list(BOREDOM)
    pts = _pts(a)
    dp = [0] * len(pts)
    steps = [
        row_step(
            "pts[1] = 1 · pts[2] = 4",
            [("pts", pts[:], None)],
            eq="pts[v] = v · count(v)",
        ),
        row_step(
            "dp[0] = 0",
            [("pts", pts[:], None), ("dp", dp[:], 0)],
            eq="dp[0] = 0",
        ),
    ]
    for v in range(1, len(pts)):
        earlier = dp[v - 2] if v >= 2 else 0
        dp[v] = max(dp[v - 1], pts[v] + earlier)
        steps.append(
            row_step(
                f"dp[{v}] = {dp[v]}",
                [("pts", pts[:], v), ("dp", dp[:], v)],
                eq=f"{dp[v]} = max({dp[v - 1]}, {pts[v]}+{earlier})",
            )
        )
    return "Boredom — tabulation", f"a = {a}", steps, show(dp[-1])


def _boredom_opt():
    a = list(BOREDOM)
    pts = _pts(a)
    prev2, prev1 = 0, 0
    steps = [
        chip_step(
            "pts[1] = 1 · pts[2] = 4",
            [("pts[1]", pts[1], "amber"), ("pts[2]", pts[2], "amber")],
            eq="pts[v] = v · count(v)",
        ),
        chip_step(
            "prev2, prev1 = 0, 0",
            _pair(prev2, prev1),
            eq="prev2, prev1 = 0, 0",
        ),
    ]
    for v in range(1, len(pts)):
        old2, old1 = prev2, prev1
        prev2, prev1 = old1, max(old1, pts[v] + old2)
        steps.append(
            chip_step(
                f"v = {v}",
                _pair(prev2, prev1) + [(f"pts[{v}]", pts[v], "amber")],
                eq=f"{prev1} = max({old1}, {pts[v]}+{old2})",
            )
        )
    return "Boredom — optimal", f"a = {a}", steps, show(prev1)


def stories():
    return {
        ("robber", "rec"): _robber_rec(),
        ("robber", "memo"): _robber_memo(),
        ("robber", "tab"): _robber_tab(),
        ("robber", "opt1"): _robber_opt1(),
        ("robber", "opt2"): _robber_opt2(),
        ("boredom", "rec"): _boredom_rec(),
        ("boredom", "memo"): _boredom_memo(),
        ("boredom", "tab"): _boredom_tab(),
        ("boredom", "opt"): _boredom_opt(),
    }


if __name__ == "__main__":
    data = stories()
    keys = [
        ("robber", "rec"),
        ("robber", "memo"),
        ("robber", "tab"),
        ("robber", "opt1"),
        ("robber", "opt2"),
        ("boredom", "rec"),
        ("boredom", "memo"),
        ("boredom", "tab"),
        ("boredom", "opt"),
    ]
    assert list(data) == keys
    for key in keys:
        title, line, steps, final = data[key]
        assert 3 <= len(steps) <= 8, (key, len(steps), [s["title"] for s in steps])
        assert all(not s["title"].startswith("Step") for s in steps)
        print(key, final)
    assert data[("robber", "rec")][3] == "11"
    assert data[("robber", "memo")][3] == "11"
    assert data[("robber", "tab")][3] == "11"
    assert data[("robber", "opt1")][3] == "11"
    assert data[("robber", "opt2")][3].startswith("11")
    for form in ("rec", "memo", "tab", "opt"):
        assert data[("boredom", form)][3] == "4"
    memo_title, _, memo_steps, _ = data[("robber", "memo")]
    assert "memo" in memo_title or any("memo" in s["title"] for s in memo_steps)
    print("PASS")
