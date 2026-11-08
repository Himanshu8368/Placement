"""Step-board data for unique paths and minimum path sum. No drawing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import run_call, row_step, grid_step, chip_step, show
import math

# The 3×3 unique-paths sample never revisits a stored cell before the answer.
# Plain recursion does repeat work: (1,1) is entered twice and (2,2) is entered
# twice. Memo returns both before the store — (1,1) is the obstacle and (2,2)
# is the goal — so the second entry is a base, not log.hit. No step title
# contains "memo". The chapter example is kept anyway.
# The 2×2 minimum-path sample is the same kind of tree: the repeated cell is
# the goal (1,1), which returns before the store.

_UNIQUE = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
_UNIQUE_LINE = "grid = [[0,0,0],[0,1,0],[0,0,0]]"
_MIN = [[1, 3], [1, 5]]
_MIN_LINE = "g = [[1, 3], [1, 5]]"
_OBSTACLE = {(1, 1): "red"}


def _label(r, c):
    return f"({r},{c})"


def _path_str(path):
    return " → ".join(_label(r, c) for r, c in path)


def _unique_rec():
    grid = _UNIQUE

    def build(log):
        R, C = len(grid), len(grid[0])

        def go(r, c):
            log.call(_label(r, c))
            if r >= R or c >= C or grid[r][c]:
                return log.base(0)
            if (r, c) == (R - 1, C - 1):
                return log.base(1)
            return log.ret(go(r + 1, c) + go(r, c + 1))

        return go(0, 0)

    answer, steps = run_call(build)
    return ("Unique paths — recursion", _UNIQUE_LINE, steps, show(answer))


def _unique_memo():
    grid = _UNIQUE

    def build(log):
        R, C = len(grid), len(grid[0])
        memo = [[-1] * C for _ in range(R)]

        def go(r, c):
            # Out of bounds, the obstacle, and the goal return before any store.
            if 0 <= r < R and 0 <= c < C and memo[r][c] >= 0:
                return log.hit(_label(r, c), memo[r][c])
            log.call(_label(r, c))
            if r >= R or c >= C or grid[r][c]:
                return log.base(0)
            if (r, c) == (R - 1, C - 1):
                return log.base(1)
            memo[r][c] = go(r + 1, c) + go(r, c + 1)
            return log.ret(memo[r][c])

        return go(0, 0)

    answer, steps = run_call(build)
    return ("Unique paths — memo", _UNIQUE_LINE, steps, show(answer))


def _unique_tab():
    grid = _UNIQUE
    R, C = len(grid), len(grid[0])
    dp = [[0] * C for _ in range(R)]
    # Nine events; keep eight. (0,1) is omitted as a frame and is already 1
    # in the following snapshot. Start, the obstacle, and the tail stay.
    skip_frame = {(0, 1)}
    steps = []

    def snap(r, c, eq, written):
        if (r, c) in skip_frame:
            return
        steps.append(grid_step(
            eq, dp, hi=(r, c) if written else None, marks=dict(_OBSTACLE), eq=eq,
        ))

    if grid[0][0] == 0:
        dp[0][0] = 1
        snap(0, 0, "dp[0][0] = 1", True)
    for r in range(R):
        for c in range(C):
            if grid[r][c] or (r, c) == (0, 0):
                if grid[r][c]:
                    snap(r, c, f"{_label(r, c)} = 0", False)
                continue
            up = dp[r - 1][c] if r else 0
            left = dp[r][c - 1] if c else 0
            dp[r][c] = up + left
            snap(r, c, f"dp[{r}][{c}] = {up} + {left}", True)
    return ("Unique paths — tabulation", _UNIQUE_LINE, steps, show(dp[-1][-1]))


def _unique_opt():
    grid = _UNIQUE
    m = len(grid[0])
    row = [0] * m
    row[0] = 1
    steps = [row_step("row[0] = 1", [("row", row[:], 0)], eq="row[0] = 1")]
    last_eq = ""
    for r, line in enumerate(grid):
        for c in range(m):
            if line[c]:
                row[c] = 0
                last_eq = f"{_label(r, c)} = 0"
            elif c:
                above = row[c]
                left = row[c - 1]
                row[c] = above + left
                last_eq = f"{row[c]} = {above} + {left}"
            else:
                last_eq = f"row[0] = {row[0]}"
        title = f"row = [{', '.join(show(v) for v in row)}]"
        hi = m - 1
        steps.append(row_step(title, [("row", row[:], hi)], eq=last_eq))
    return ("Unique paths — optimal", _UNIQUE_LINE, steps, show(row[-1]))


def _min_fill(g):
    """Full table and the walk back. Returns dp, per-cell (r, c, eq), forward path."""
    R, C = len(g), len(g[0])
    dp = [[0] * C for _ in range(R)]
    cells = []
    for r in range(R):
        for c in range(C):
            if r == 0 and c == 0:
                dp[r][c] = g[0][0]
                eq = f"dp[0][0] = {g[0][0]}"
            else:
                up = dp[r - 1][c] if r else math.inf
                left = dp[r][c - 1] if c else math.inf
                dp[r][c] = g[r][c] + min(up, left)
                eq = f"{show(dp[r][c])} = {g[r][c]} + min({show(up)}, {show(left)})"
            cells.append((r, c, eq))
    r, c = R - 1, C - 1
    back = [(r, c)]
    while (r, c) != (0, 0):
        if r and dp[r - 1][c] == dp[r][c] - g[r][c]:
            r -= 1
        else:
            c -= 1
        back.append((r, c))
    return dp, cells, back[::-1]


def _minpath_rec():
    g = _MIN

    def build(log):
        R, C = len(g), len(g[0])

        def f(r, c):
            log.call(_label(r, c))
            if r == R or c == C:
                return log.base(math.inf)
            if r == R - 1 and c == C - 1:
                return log.base(g[r][c])
            return log.ret(g[r][c] + min(f(r + 1, c), f(r, c + 1)))

        return f(0, 0)

    answer, steps = run_call(build)
    return ("Minimum path sum — recursion", _MIN_LINE, steps, show(answer))


def _minpath_memo():
    g = _MIN

    def build(log):
        R, C = len(g), len(g[0])
        memo = [[None] * C for _ in range(R)]

        def f(r, c):
            if 0 <= r < R and 0 <= c < C and memo[r][c] is not None:
                return log.hit(_label(r, c), memo[r][c])
            log.call(_label(r, c))
            if r == R or c == C:
                return log.base(math.inf)
            if r == R - 1 and c == C - 1:
                return log.base(g[r][c])
            memo[r][c] = g[r][c] + min(f(r + 1, c), f(r, c + 1))
            return log.ret(memo[r][c])

        return f(0, 0)

    answer, steps = run_call(build)
    return ("Minimum path sum — memo", _MIN_LINE, steps, show(answer))


def _minpath_tab():
    g = _MIN
    dp, cells, path = _min_fill(g)
    R, C = len(g), len(g[0])
    shown = [["·"] * C for _ in range(R)]
    steps = []
    for r, c, eq in cells:
        shown[r][c] = dp[r][c]
        title = f"dp[{r}][{c}] = {show(dp[r][c])}"
        frame_eq = eq
        if (r, c) == (R - 1, C - 1):
            frame_eq = f"{eq} · {_path_str(path)}"
        steps.append(grid_step(title, shown, hi=(r, c), eq=frame_eq))
    return ("Minimum path sum — tabulation", _MIN_LINE, steps, show(dp[-1][-1]))


def _minpath_opt1():
    g = _MIN
    num = [0] * len(g[0])
    steps = []
    for r, line in enumerate(g):
        for c, x in enumerate(line):
            if r == 0 and c == 0:
                num[c] = x
                eq = f"row[0] = {x}"
            else:
                up = num[c] if r else math.inf
                left = num[c - 1] if c else math.inf
                num[c] = x + min(up, left)
                eq = f"{show(num[c])} = {x} + min(above {show(up)}, left {show(left)})"
            display = [num[i] if (r > 0 or i <= c) else "·" for i in range(len(num))]
            steps.append(row_step(
                f"row[{c}] = {show(num[c])}", [("row", display, c)], eq=eq,
            ))
    return ("Minimum path sum — optimal 1", _MIN_LINE, steps, show(num[-1]))


def _minpath_opt2():
    g = _MIN
    dp, _cells, path = _min_fill(g)
    R, C = len(g), len(g[0])
    filled = [row[:] for row in dp]
    steps = [grid_step(
        f"dp[{R - 1}][{C - 1}] = {show(dp[-1][-1])}",
        filled, hi=(R - 1, C - 1), eq="filled table",
    )]
    r, c = R - 1, C - 1
    while (r, c) != (0, 0):
        need = dp[r][c] - g[r][c]
        if r and dp[r - 1][c] == need:
            nr, nc = r - 1, c
            via = "above"
        else:
            nr, nc = r, c - 1
            via = "left"
        steps.append(grid_step(
            f"{_label(nr, nc)} = {show(dp[nr][nc])}",
            filled,
            hi=(nr, nc),
            eq=f"{show(dp[r][c])} − {g[r][c]} = {show(need)}, {via}",
        ))
        r, c = nr, nc
    final = f"{show(dp[-1][-1])} · {_path_str(path)}"
    return ("Minimum path sum — optimal 2", _MIN_LINE, steps, final)


def stories() -> dict:
    return {
        ("unique", "rec"): _unique_rec(),
        ("unique", "memo"): _unique_memo(),
        ("unique", "tab"): _unique_tab(),
        ("unique", "opt"): _unique_opt(),
        ("minpath", "rec"): _minpath_rec(),
        ("minpath", "memo"): _minpath_memo(),
        ("minpath", "tab"): _minpath_tab(),
        ("minpath", "opt1"): _minpath_opt1(),
        ("minpath", "opt2"): _minpath_opt2(),
    }


if __name__ == "__main__":
    # chip_step is part of the shared step-board API used by this chapter.
    assert callable(chip_step)
    data = stories()
    order = [
        ("unique", "rec"), ("unique", "memo"), ("unique", "tab"), ("unique", "opt"),
        ("minpath", "rec"), ("minpath", "memo"), ("minpath", "tab"),
        ("minpath", "opt1"), ("minpath", "opt2"),
    ]
    assert list(data) == order
    for key in order:
        title, line, steps, final = data[key]
        assert 3 <= len(steps) <= 8, (key, len(steps), [s["title"] for s in steps])
        assert all(not s["title"].startswith("Step") for s in steps), key
        assert "=" in steps[-1]["title"], (key, steps[-1]["title"])
    assert data[("unique", "rec")][1] == _UNIQUE_LINE
    assert data[("minpath", "rec")][1] == _MIN_LINE
    for form in ("rec", "memo", "tab", "opt"):
        assert data[("unique", form)][3] == "2", (form, data[("unique", form)][3])
    for form in ("rec", "memo", "tab", "opt1"):
        assert data[("minpath", form)][3] == "7", (form, data[("minpath", form)][3])
    assert data[("minpath", "opt2")][3].startswith("7")
    assert data[("minpath", "opt2")][3] == "7 · (0,0) → (1,0) → (1,1)"
    tab_steps = data[("unique", "tab")][2]
    assert tab_steps[-1]["grid"] == [["1", "1", "1"], ["1", "0", "1"], ["1", "1", "2"]]
    assert all(s["marks"][(1, 1)] == "red" for s in tab_steps)
    assert data[("unique", "opt")][2][-1]["eq"] == "2 = 1 + 1"
    memo_titles = [s["title"] for s in data[("unique", "memo")][2]]
    # No stored cell is entered twice (see the module comment), so a "memo"
    # title is not required. A hit, if a different input produced one, would.
    if any("memo" in t for t in memo_titles):
        assert any(t.startswith("memo") for t in memo_titles)
    print("keys:", [f"{p}/{f}" for p, f in order])
    print("finals:", {f"{p}/{f}": data[(p, f)][3] for p, f in order})
    print("steps:", {f"{p}/{f}": len(data[(p, f)][2]) for p, f in order})
    print("PASS")
