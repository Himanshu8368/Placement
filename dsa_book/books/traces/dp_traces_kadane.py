"""Step boards for three Kadane-style DP problems. Data only; drawing lives elsewhere."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import CallLog, run_call, row_step, chip_step, line_step, show
import math

_MINUS = "\u2212"  # same minus sign show() uses for −∞


def _fmt(xs):
    return "[" + ", ".join(show(x) for x in xs) + "]"


def _replay(build):
    """run_call checks that build can run twice. Frames come from the same rule."""
    answer, _sampled = run_call(build)
    log = CallLog(collapse=False)
    got = build(log)
    if log.too_wide():
        log = CallLog(collapse=True)
        got = build(log)
    assert got == answer
    return answer, log.frames


def _take(frames, titles):
    """Pull those return / base / memo snapshots, in the order the titles occur."""
    marked = [f for f in frames if "=" in f.get("title", "")]
    used = set()
    chosen = []
    for title in titles:
        for i, frame in enumerate(marked):
            if i in used or frame["title"] != title:
                continue
            used.add(i)
            chosen.append(frame)
            break
        else:
            raise AssertionError("missing step " + title)
    return chosen


def _finish(answer, frames, titles, eq):
    steps = _take(frames, titles)
    assert titles[-1].endswith(" = " + show(answer))
    steps[-1]["eq"] = eq
    return show(answer), steps


def _deletion_end_label(i, deleted):
    return f"e{i}.{'D' if deleted else 'K'}"


def _deletion_rec():
    a = [1, -2, 3]

    def build(log):
        def end(i, deleted):
            log.call(_deletion_end_label(i, deleted))
            if i < 0:
                return log.base(-math.inf)
            if not deleted:
                return log.ret(max(a[i], a[i] + end(i - 1, False)))
            return log.ret(max(a[i] + end(i - 1, True), end(i - 1, False)))

        log.call("best")
        best = -math.inf
        for i in range(len(a)):
            best = max(best, end(i, False), end(i, True))
        return log.ret(best)

    inf = show(-math.inf)
    return _finish(*_replay(build), [
        f"e-1.K = {inf}",
        "e0.K = 1",
        f"e0.D = {inf}",
        "e1.K = -1",
        "e1.D = 1",
        "e2.K = 3",
        "e2.D = 4",
        "best = 4",
    ], "answer = 4")


def _deletion_memo():
    a = [1, -2, 3]

    def build(log):
        memo = {}

        def end(i, deleted):
            label = _deletion_end_label(i, deleted)
            if i < 0:
                log.call(label)
                return log.base(-math.inf)
            key = (i, deleted)
            if key in memo:
                return log.hit(label, memo[key])
            log.call(label)
            if not deleted:
                val = max(a[i], a[i] + end(i - 1, False))
            else:
                val = max(a[i] + end(i - 1, True), end(i - 1, False))
            memo[key] = val
            return log.ret(val)

        log.call("best")
        best = -math.inf
        for i in range(len(a)):
            best = max(best, end(i, False), end(i, True))
        return log.ret(best)

    inf = show(-math.inf)
    return _finish(*_replay(build), [
        f"e-1.K = {inf}",
        "e0.K = 1",
        f"e0.D = {inf}",
        "memo e0.K = 1",
        "e1.D = 1",
        "memo e1.D = 1",
        "e2.D = 4",
        "best = 4",
    ], "answer = 4")


def _deletion_tab():
    a = [1, -2, 3]
    keep, dele = [a[0]], [-math.inf]
    steps = [
        row_step(
            f"keep = {_fmt(keep)}, dele = {_fmt(dele)}",
            [("keep", list(keep), 0), ("dele", list(dele), 0)],
            eq=f"keep[0] = a[0], dele[0] = {_MINUS}∞",
        )
    ]
    for i in range(1, len(a)):
        keep.append(max(keep[-1] + a[i], a[i]))
        dele.append(max(dele[-1] + a[i], keep[-2]))
        steps.append(
            row_step(
                f"keep = {_fmt(keep)}, dele = {_fmt(dele)}",
                [("keep", list(keep), i), ("dele", list(dele), i)],
                eq=f"x = {show(a[i])}; dele reads keep[-2], the old keep",
            )
        )
    ans = max(max(keep), max(dele))
    steps.append(
        row_step(
            f"answer = {show(ans)}",
            [("keep", list(keep), keep.index(max(keep))), ("dele", list(dele), dele.index(ans))],
            eq=f"max(max(keep), max(dele)) = {show(ans)}",
        )
    )
    return show(ans), steps


def _deletion_opt():
    a = [1, -2, 3]
    keep, dele, best = a[0], -math.inf, a[0]
    steps = [
        chip_step(
            f"keep = {show(keep)}, dele = {show(dele)}, best = {show(best)}",
            [("keep", keep, "indigo"), ("dele", dele, "pink"), ("best", best, "green")],
            eq="dele reads the old keep",
        )
    ]
    for x in a[1:]:
        keep, dele = max(keep + x, x), max(dele + x, keep)
        best = max(best, keep, dele)
        steps.append(
            chip_step(
                f"x = {show(x)}, keep = {show(keep)}, dele = {show(dele)}, best = {show(best)}",
                [("keep", keep, "indigo"), ("dele", dele, "pink"), ("best", best, "green")],
                eq="keep, dele = max(keep + x, x), max(dele + x, keep); dele reads the old keep",
            )
        )
    steps.append(
        line_step(
            f"answer = {show(best)}",
            [f"keep = {show(keep)}", f"dele = {show(dele)}", f"best = {show(best)}"],
            eq="dele reads the old keep",
        )
    )
    return show(best), steps


def _beautiful_go(a, x, log, memo=None):
    n = len(a)

    def go(i, phase):
        label = f"{i}.{phase}"
        if i == n:
            log.call(label)
            return log.base(0)
        if memo is not None and (i, phase) in memo:
            return log.hit(label, memo[(i, phase)])
        log.call(label)
        v = a[i]
        if phase == 0:
            val = max(go(i + 1, 0), v + go(i + 1, 1), x * v + go(i + 1, 2))
        elif phase == 1:
            val = max(0, v + go(i + 1, 1), x * v + go(i + 1, 2))
        elif phase == 2:
            val = max(0, x * v + go(i + 1, 2), v + go(i + 1, 3))
        else:
            val = max(0, v + go(i + 1, 3))
        if memo is not None:
            memo[(i, phase)] = val
        return log.ret(val)

    return go(0, 0)


def _beautiful_rec():
    a, x = [-3, 8], -2

    def build(log):
        return _beautiful_go(a, x, log)

    return _finish(*_replay(build), [
        "2.0 = 0",
        "2.1 = 0",
        "2.2 = 0",
        "1.0 = 8",
        "1.1 = 8",
        "2.3 = 0",
        "1.2 = 8",
        "0.0 = 14",
    ], "go(0, 0) = 14")


def _beautiful_memo():
    a, x = [-3, 8], -2

    def build(log):
        return _beautiful_go(a, x, log, memo={})

    return _finish(*_replay(build), [
        "2.0 = 0",
        "2.1 = 0",
        "2.2 = 0",
        "1.0 = 8",
        "1.1 = 8",
        "2.3 = 0",
        "1.2 = 8",
        "0.0 = 14",
    ], "go(0, 0) = 14; i = n is a base and is not stored")


def _beautiful_tab():
    a, x = [-3, 8, -2, 1, -6], -2
    n = len(a)
    dp = [[0] * 4 for _ in range(n + 1)]
    steps = []
    for i in range(n - 1, -1, -1):
        v = a[i]
        dp[i][3] = max(0, v + dp[i + 1][3])
        dp[i][2] = max(0, x * v + dp[i + 1][2], v + dp[i + 1][3])
        dp[i][1] = max(0, v + dp[i + 1][1], x * v + dp[i + 1][2])
        dp[i][0] = max(dp[i + 1][0], v + dp[i + 1][1], x * v + dp[i + 1][2])
        eq = f"v = {show(v)}"
        if i == 0:
            eq = f"dp[0][0] = {show(dp[0][0])}"
        steps.append(
            row_step(
                f"dp[{i}] = {_fmt(dp[i])}",
                [("phase", list(dp[i]), 0), ("next", list(dp[i + 1]), None)],
                eq=eq,
            )
        )
    return show(dp[0][0]), steps


def _beautiful_opt():
    a, x = [-3, 8, -2, 1, -6], -2
    d0 = d1 = d2 = best = 0
    steps = []
    for v in a:
        d0 = max(0, d0 + v)
        d1 = max(d0, d1 + x * v)
        d2 = max(d1, d2 + v)
        best = max(best, d2)
        steps.append(
            chip_step(
                f"v = {show(v)}, (d0, d1, d2) = ({show(d0)}, {show(d1)}, {show(d2)})",
                [
                    ("d0", d0, "indigo"),
                    ("d1", d1, "amber"),
                    ("d2", d2, "green"),
                    ("best", best, "cyan"),
                ],
                eq="d1 reads the new d0; d2 reads the new d1",
            )
        )
    return show(best), steps


def _circular_ends(a, log, mx=None, mn=None):
    def end_max(i):
        label = f"M{i}"
        if i == 0:
            log.call(label)
            return log.base(a[0])
        if mx is not None and i in mx:
            return log.hit(label, mx[i])
        log.call(label)
        val = max(a[i], a[i] + end_max(i - 1))
        if mx is not None:
            mx[i] = val
        return log.ret(val)

    def end_min(i):
        label = f"m{i}"
        if i == 0:
            log.call(label)
            return log.base(a[0])
        if mn is not None and i in mn:
            return log.hit(label, mn[i])
        log.call(label)
        val = min(a[i], a[i] + end_min(i - 1))
        if mn is not None:
            mn[i] = val
        return log.ret(val)

    return end_max, end_min


def _circular_answer(a, end_max, end_min):
    big = max(end_max(i) for i in range(len(a)))
    small = min(end_min(i) for i in range(len(a)))
    total = sum(a)
    return big if big < 0 else max(big, total - small)


def _circular_rec():
    a = [5, -3, 5]

    def build(log):
        end_max, end_min = _circular_ends(a, log)
        log.call("ans")
        return log.ret(_circular_answer(a, end_max, end_min))

    return _finish(*_replay(build), [
        "M0 = 5",
        "M0 = 5",
        "M1 = 2",
        "M2 = 7",
        "m0 = 5",
        "m1 = -3",
        "m2 = 2",
        "ans = 10",
    ], f"max(7, 7{_MINUS}({_MINUS}3)) = 10")


def _circular_memo():
    a = [5, -3, 5]

    def build(log):
        end_max, end_min = _circular_ends(a, log, mx={}, mn={})
        log.call("ans")
        return log.ret(_circular_answer(a, end_max, end_min))

    return _finish(*_replay(build), [
        "M0 = 5",
        "M0 = 5",
        "M1 = 2",
        "memo M1 = 2",
        "M2 = 7",
        "m1 = -3",
        "memo m1 = -3",
        "ans = 10",
    ], f"max(7, 7{_MINUS}({_MINUS}3)) = 10")


def _circular_tab():
    a = [5, -3, 5]
    mx, mn = [a[0]], [a[0]]
    steps = [
        row_step(
            f"mx = {_fmt(mx)}, mn = {_fmt(mn)}",
            [("mx", list(mx), 0), ("mn", list(mn), 0)],
            eq="mx[0] = mn[0] = a[0]",
        )
    ]
    for i in range(1, len(a)):
        mx.append(max(a[i], mx[-1] + a[i]))
        mn.append(min(a[i], mn[-1] + a[i]))
        steps.append(
            row_step(
                f"mx = {_fmt(mx)}, mn = {_fmt(mn)}",
                [("mx", list(mx), i), ("mn", list(mn), i)],
                eq=f"a[{i}] = {show(a[i])}",
            )
        )
    big, small, total = max(mx), min(mn), sum(a)
    ans = big if big < 0 else max(big, total - small)
    steps.append(
        line_step(
            f"max(7, 7{_MINUS}({_MINUS}3)) = {show(ans)}",
            [f"M = {show(big)}", f"m = {show(small)}", f"S = {show(total)}"],
            eq=f"max(7, 7{_MINUS}({_MINUS}3)) = 10",
        )
    )
    return show(ans), steps


def _circular_opt():
    a = [5, -3, 5]
    cur_max = cur_min = 0
    best_max, best_min, total = -math.inf, math.inf, 0
    steps = []
    for i, v in enumerate(a):
        cur_max = max(v, cur_max + v)
        cur_min = min(v, cur_min + v)
        best_max, best_min, total = max(best_max, cur_max), min(best_min, cur_min), total + v
        last = i == len(a) - 1
        steps.append(
            chip_step(
                f"v = {show(v)}, best_max = {show(best_max)}, best_min = {show(best_min)}, total = {show(total)}",
                [
                    ("cur_max", cur_max, "indigo"),
                    ("cur_min", cur_min, "pink"),
                    ("best_max", best_max, "green"),
                    ("best_min", best_min, "amber"),
                    ("total", total, "cyan"),
                ],
                eq=f"max(7, 7{_MINUS}({_MINUS}3)) = 10" if last else "cur_max and cur_min extend or restart",
            )
        )
    ans = best_max if best_max < 0 else max(best_max, total - best_min)
    return show(ans), steps


def stories():
    del_rec, del_memo = _deletion_rec(), _deletion_memo()
    del_tab, del_opt = _deletion_tab(), _deletion_opt()
    beau_rec, beau_memo = _beautiful_rec(), _beautiful_memo()
    beau_tab, beau_opt = _beautiful_tab(), _beautiful_opt()
    circ_rec, circ_memo = _circular_rec(), _circular_memo()
    circ_tab, circ_opt = _circular_tab(), _circular_opt()
    return {
        ("deletion", "rec"): ("One deletion — recursion", "a = [1, -2, 3]", del_rec[1], del_rec[0]),
        ("deletion", "memo"): ("One deletion — memo", "a = [1, -2, 3]", del_memo[1], del_memo[0]),
        ("deletion", "tab"): ("One deletion — tabulation", "a = [1, -2, 3]", del_tab[1], del_tab[0]),
        ("deletion", "opt"): ("One deletion — optimal", "a = [1, -2, 3]", del_opt[1], del_opt[0]),
        ("beautiful", "rec"): ("Beautiful array — recursion", "a = [-3, 8], x = -2", beau_rec[1], beau_rec[0]),
        ("beautiful", "memo"): ("Beautiful array — memo", "a = [-3, 8], x = -2", beau_memo[1], beau_memo[0]),
        ("beautiful", "tab"): (
            "Beautiful array — tabulation",
            "a = [-3, 8, -2, 1, -6], x = -2",
            beau_tab[1],
            beau_tab[0],
        ),
        ("beautiful", "opt"): (
            "Beautiful array — optimal",
            "a = [-3, 8, -2, 1, -6], x = -2",
            beau_opt[1],
            beau_opt[0],
        ),
        ("circular", "rec"): ("Circular subarray — recursion", "a = [5, -3, 5]", circ_rec[1], circ_rec[0]),
        ("circular", "memo"): ("Circular subarray — memo", "a = [5, -3, 5]", circ_memo[1], circ_memo[0]),
        ("circular", "tab"): ("Circular subarray — tabulation", "a = [5, -3, 5]", circ_tab[1], circ_tab[0]),
        ("circular", "opt"): ("Circular subarray — optimal", "a = [5, -3, 5]", circ_opt[1], circ_opt[0]),
    }


if __name__ == "__main__":
    board = stories()
    again = stories()
    forms = ("rec", "memo", "tab", "opt")
    for form in forms:
        assert board[("deletion", form)][3] == "4"
        assert board[("circular", form)][3] == "10"
        assert again[("deletion", form)][3] == "4"
    assert board[("beautiful", "rec")][3] == "14"
    assert board[("beautiful", "memo")][3] == "14"
    assert board[("beautiful", "tab")][3] == "22"
    assert board[("beautiful", "opt")][3] == "22"

    inf = show(-math.inf)
    assert inf == "\u2212\u221e"
    tab_cells = [c for st in board[("deletion", "tab")][2] for row in st["rows"] for c in row["cells"]]
    opt_vals = [c["value"] for st in board[("deletion", "opt")][2] for c in st.get("chips", [])]
    assert inf in tab_cells and inf in opt_vals
    assert any("dele reads the old keep" in st.get("eq", "") for st in board[("deletion", "opt")][2])
    assert any(st["title"].startswith("memo ") for st in board[("deletion", "memo")][2])
    assert any(st["title"].startswith("memo ") for st in board[("circular", "memo")][2])

    beau_opt_steps = board[("beautiful", "opt")][2]
    triples = []
    for st in beau_opt_steps:
        chips = {c["name"]: c["value"] for c in st["chips"]}
        triples.append((chips["d0"], chips["d1"], chips["d2"]))
    assert triples == [("0", "6", "6"), ("8", "8", "14"), ("6", "12", "12"), ("7", "10", "13"), ("1", "22", "22")]
    assert any(c["value"] == "22" for st in beau_opt_steps for c in st["chips"])

    beau_tab = board[("beautiful", "tab")][2]
    assert [st["rows"][0]["cells"] for st in beau_tab] == [
        ["12", "12", "12", "0"],
        ["13", "13", "10", "1"],
        ["14", "14", "14", "0"],
        ["22", "22", "8", "8"],
        ["22", "19", "14", "5"],
    ]

    circ_rows = [st for st in board[("circular", "tab")][2] if "rows" in st]
    assert circ_rows[-1]["rows"][0]["cells"] == ["5", "2", "7"]
    assert circ_rows[-1]["rows"][1]["cells"] == ["5", "-3", "2"]
    assert board[("circular", "tab")][2][-1]["eq"] == f"max(7, 7{_MINUS}({_MINUS}3)) = 10"

    last_opt = board[("circular", "opt")][2][-1]
    last_chips = {c["name"]: c["value"] for c in last_opt["chips"]}
    assert last_chips["best_max"] == "7" and last_chips["best_min"] == "-3" and last_chips["total"] == "7"

    expected_line = {
        "deletion": "a = [1, -2, 3]",
        "circular": "a = [5, -3, 5]",
    }
    for key, (title, line, steps, final) in board.items():
        if key[0] == "beautiful":
            want = "a = [-3, 8], x = -2" if key[1] in ("rec", "memo") else "a = [-3, 8, -2, 1, -6], x = -2"
        else:
            want = expected_line[key[0]]
        assert line == want
        assert title
        assert 3 <= len(steps) <= 8, (key, len(steps), [s["title"] for s in steps])
        for st in steps:
            assert "=" in st["title"], (key, st["title"])
            assert not st["title"].startswith("Step")
        if key[1] in ("rec", "memo"):
            assert all("tree" in st for st in steps), key
        print(key, final)
    print("PASS")
