"""Step frames for the depth-first search chapter.

Drawing lives in ``dsa_viz.story``. This module only records frames.
``get(problem, form)`` returns ``(title, input line, steps, final)``.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import chip_step, grid_step, line_step, row_step, run_call, show


def _trim(steps, limit=8):
    steps = list(steps)
    if len(steps) <= limit:
        return steps
    if limit <= 1:
        return steps[:limit]
    rest = list(range(1, len(steps) - 1))
    keep = limit - 2
    if keep <= 0 or not rest:
        return [steps[0], steps[-1]][:limit]
    step = len(rest) / keep
    mid = [rest[min(len(rest) - 1, int((i + 0.5) * step))] for i in range(keep)]
    idx = sorted({0, len(steps) - 1, *mid})
    return [steps[i] for i in idx]


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    frames = _trim(steps)
    assert 1 <= len(frames) <= 8, (title, len(frames))
    return (title, inp, frames, text)


def _sp(xs):
    return " ".join(show(x) for x in xs)


def _children(n, parents, start=2):
    ch = [[] for _ in range(n + 1)]
    for i, p in enumerate(parents, start=start):
        ch[p].append(i)
    return ch


def _cycle_from(u, v, parent):
    cyc = [u]
    x = u
    while x != v:
        x = parent[x]
        cyc.append(x)
    cyc.append(u)
    return cyc


def _undirected(n, edges, log=None):
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    parent = [-1] * (n + 1)
    seen = [False] * (n + 1)
    found = []

    def go(u, p):
        if log is not None:
            log.call(str(u))
        seen[u] = True
        parent[u] = p
        for v in adj[u]:
            if v == p or found:
                continue
            if seen[v]:
                found.append(_cycle_from(u, v, parent))
                if log is not None:
                    log.ret(_sp(found[0]), "cycle " + _sp(found[0]))
                return
            go(v, u)
            if found:
                return
        if log is not None:
            log.ret("·", f"{u} back")

    for s in range(1, n + 1):
        if not seen[s] and not found:
            go(s, -1)
    return found[0] if found else None


def _directed(n, edges, log=None):
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
    color = [0] * (n + 1)
    parent = [-1] * (n + 1)
    found = []

    def go(u):
        if log is not None:
            log.call(str(u))
        color[u] = 1
        closed_here = False
        for v in adj[u]:
            if found:
                break
            if color[v] == 1:
                cyc = [u]
                x = u
                while x != v:
                    x = parent[x]
                    cyc.append(x)
                cyc.reverse()
                cyc.append(cyc[0])
                found.append(cyc)
                closed_here = True
                break
            if color[v] == 0:
                parent[v] = u
                go(v)
        if log is not None and log.stack and log.stack[-1]["label"] == str(u):
            if closed_here:
                log.ret(_sp(found[0]), "cycle " + _sp(found[0]))
            elif not found:
                log.ret("black", f"{u} black")
            # A cycle found deeper leaves this frame on the root path.
        if not found:
            color[u] = 2

    for s in range(1, n + 1):
        if color[s] == 0 and not found:
            go(s)
    return found[0] if found else None


TRIP = [(1, 3), (1, 2), (5, 3), (1, 5), (2, 4), (4, 5)]
FLIGHTS = [(1, 3), (2, 1), (2, 4), (3, 2), (3, 4)]
DAG = [(1, 2), (2, 3), (1, 3)]


def _trip_forms():
    def rec(log):
        cyc = _undirected(5, TRIP, log)
        assert cyc[0] == cyc[-1] and len(cyc) >= 4
        return _sp(cyc)

    ans, steps = run_call(rec)
    assert ans == "5 3 1 5"

    def stack_steps():
        n, edges = 5, TRIP
        adj = [[] for _ in range(n + 1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        parent = [-1] * (n + 1)
        it = [0] * (n + 1)
        seen = [False] * (n + 1)
        frames = []
        for s in range(1, n + 1):
            if seen[s]:
                continue
            seen[s] = True
            st = [s]
            frames.append(line_step(f"start {s}", [f"stack {_sp(st)}"]))
            while st:
                u = st[-1]
                if it[u] == len(adj[u]):
                    st.pop()
                    continue
                v = adj[u][it[u]]
                it[u] += 1
                if v == parent[u]:
                    continue
                if seen[v]:
                    cyc = _cycle_from(u, v, parent)
                    frames.append(line_step(
                        f"{u} sees {v}",
                        [f"stack {_sp(st)}", f"cycle {_sp(cyc)}"],
                        "visited, and not the parent",
                    ))
                    return cyc, frames
                parent[v] = u
                seen[v] = True
                st.append(v)
                frames.append(line_step(f"enter {v}", [f"stack {_sp(st)}", f"parent {v} = {u}"]))
        return None, frames

    cyc, frames = stack_steps()
    assert _sp(cyc) == "5 3 1 5"
    out = {}
    out[("trip", "rec")] = _pack("Round trip — recursion", "5 cities", steps, ans)
    out[("trip", "stack")] = _pack("Round trip — cursor stack", "5 cities", frames, _sp(cyc))
    return out


def _trip2_forms():
    def colours(log):
        cyc = _directed(4, FLIGHTS, log)
        return _sp(cyc)

    ans, steps = run_call(colours)
    assert ans == "1 3 2 1"

    def plain(log):
        cyc = _undirected(3, DAG, log)
        return _sp(cyc)

    false, frames = run_call(plain)
    assert false == "3 2 1 3"
    assert _directed(3, DAG) is None
    frames = list(frames) + [line_step(
        "three colours on the same arcs",
        ["1 → 3 lands on black", "no gray head"],
        "directed answer: none",
    )]
    return {
        ("trip2", "colors"): _pack("Round trip II — three colours", "flights 1→3, 3→2, 2→1", steps, ans),
        ("trip2", "plain"): _pack("Round trip II — ignore the colours", "DAG 1→2→3, 1→3", frames, false),
    }


def _balance(parents, s):
    n = len(s)
    bal = [0] + [1 if c == "W" else -1 for c in s]
    ans = 0
    snaps = [("start", list(bal), None, ans)]
    for v in range(n, 1, -1):
        if bal[v] == 0:
            ans += 1
        bal[parents[v - 2]] += bal[v]
        snaps.append((f"fold {v}", list(bal), v, ans))
    ans += bal[1] == 0
    snaps.append(("root", list(bal), 1, ans))
    return ans, snaps


def _wb_forms():
    parents, s = [1, 1, 2, 3, 3, 5], "WBBWWBW"
    ans, snaps = _balance(parents, s)
    assert ans == 2
    frames = []
    for title, bal, hi, count in snaps:
        frames.append(row_step(
            f"{title}, count {count}",
            [("bal", bal[1:], None if hi is None else hi - 1)],
            f"answer {count}",
        ))

    def rec(log):
        n = len(s)
        ch = _children(n, parents)
        count = [0]

        def go(u):
            log.call(str(u))
            total = 1 if s[u - 1] == "W" else -1
            for v in ch[u]:
                total += go(v)
            count[0] += total == 0
            return log.ret(total, f"{u} = {show(total)}")

        go(1)
        return count[0]

    got, tree = run_call(rec)
    assert got == 2
    tree = list(tree) + [line_step("two zeros", ["vertex 5 sums to 0", "vertex 2 sums to 0"], "answer 2")]
    return {
        ("wb", "index"): _pack("White-black — index loop", "WBBWWBW", frames, ans),
        ("wb", "rec"): _pack("White-black — recursion", "WBBWWBW", tree, got),
    }


def _sizes_post(parents):
    n = len(parents)
    ch = [[] for _ in range(n)]
    root = parents.index(-1)
    for v, p in enumerate(parents):
        if p != -1:
            ch[p].append(v)
    sz = [1] * n
    snaps = []

    def go(u):
        for v in ch[u]:
            go(v)
            sz[u] += sz[v]
        snaps.append((u, list(sz)))

    go(root)
    return sz, ch, root, snaps


def _score_from(n, ch, sz, root):
    best = count = 0
    detail = []
    for v in range(n):
        prod, parts = 1, 0
        for c in ch[v]:
            prod *= sz[c]
            parts += 1
        above = n - sz[v]
        if v != root and above:
            prod *= above
            parts += 1
        if parts == 0:
            prod = 0
        detail.append(prod)
        if prod > best:
            best, count = prod, 1
        elif prod == best:
            count += 1
    return count, detail


def _score_index(parents):
    n = len(parents)
    root = parents.index(-1)
    sz = [1] * n
    for v in range(n - 1, 0, -1):
        if parents[v] != -1:
            sz[parents[v]] += sz[v]
    ch = [[] for _ in range(n)]
    for v, p in enumerate(parents):
        if p != -1:
            ch[p].append(v)
    count, _ = _score_from(n, ch, sz, root)
    return count


def _score_forms():
    sample = [-1, 2, 0, 2, 0]
    sz, ch, root, snaps = _sizes_post(sample)
    count, detail = _score_from(len(sample), ch, sz, root)
    assert count == 3 and detail == [3, 4, 2, 4, 4]
    other = [-1, 2, 3, 0]
    sz2, ch2, root2, _ = _sizes_post(other)
    assert _score_from(len(other), ch2, sz2, root2)[0] == 2
    assert _score_index(other) == 1
    frames = []
    for u, row in snaps:
        frames.append(row_step(f"size of {u} is {row[u]}", [("sz", row, u)], f"sz[{u}] = {row[u]}"))
    frames.append(line_step(
        "index loop on a rising parent",
        ["[-1, 2, 3, 0]", "index loop returns 1", "post-order returns 2"],
        "parents[v] < v is required",
    ))
    frames.append(row_step("three nodes score 4", [("score", detail, None)], "answer 3"))

    def brute_frames():
        parents = sample
        n = len(parents)
        adj = [[] for _ in range(n)]
        for v, p in enumerate(parents):
            if p != -1:
                adj[v].append(p)
                adj[p].append(v)
        best = count = 0
        frames = []
        for cut in range(n):
            seen = [False] * n
            seen[cut] = True
            prod, parts = 1, 0
            for s in range(n):
                if seen[s]:
                    continue
                q = deque([s])
                seen[s] = True
                comp = 0
                while q:
                    u = q.popleft()
                    comp += 1
                    for v in adj[u]:
                        if not seen[v]:
                            seen[v] = True
                            q.append(v)
                prod *= comp
                parts += 1
            if parts == 0:
                prod = 0
            if prod > best:
                best, count = prod, 1
            elif prod == best:
                count += 1
            frames.append(chip_step(
                f"delete {cut}",
                [("product", prod, "indigo"), ("best", best, "amber"), ("count", count, "green")],
                f"pieces multiply to {prod}",
            ))
        return count, frames

    bcount, bframes = brute_frames()
    assert bcount == 3
    return {
        ("score", "post"): _pack("Highest score — post-order", "[-1, 2, 0, 2, 0]", frames, count),
        ("score", "brute"): _pack("Highest score — delete each node", "[-1, 2, 0, 2, 0]", bframes, bcount),
    }


def _ki_push(n, edges, ops):
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    parent = [-1] * (n + 1)
    parent[1] = 0
    order = []
    q = deque([1])
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            if v != parent[u]:
                parent[v] = u
                q.append(v)
    cnt = [0] * (n + 1)
    for p, x in ops:
        cnt[p] += x
    snaps = [("operations recorded", list(cnt))]
    for u in order:
        if u != 1:
            cnt[u] += cnt[parent[u]]
            snaps.append((f"{parent[u]} pushes to {u}", list(cnt)))
    return cnt[1:], snaps


def _ki_walk(n, edges, ops):
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    parent = [0] * (n + 1)

    def root(u, p):
        parent[u] = p
        for v in adj[u]:
            if v != p:
                root(v, u)

    root(1, 0)
    cnt = [0] * (n + 1)
    snaps = []

    def under(u, p, x):
        cnt[u] += x
        for v in adj[u]:
            if v != p:
                under(v, u, x)

    for p, x in ops:
        under(p, parent[p], x)
        snaps.append((f"add {x} under {p}", list(cnt)))
    return cnt[1:], snaps


def _ki_forms():
    edges, ops = [(1, 2), (2, 3), (2, 4)], [(2, 10), (1, 100), (3, 1)]
    pushed, psnaps = _ki_push(4, edges, ops)
    walked, wsnaps = _ki_walk(4, edges, ops)
    assert pushed == walked == [100, 110, 111, 110]
    pframes = [row_step(title, [("cnt", row[1:], None)], _sp(row[1:])) for title, row in psnaps]
    wframes = [row_step(title, [("cnt", row[1:], None)], _sp(row[1:])) for title, row in wsnaps]
    return {
        ("ki", "bfs"): _pack("Ki — BFS order", "ops (2, 10), (1, 100), (3, 1)", pframes, _sp(pushed)),
        ("ki", "walk"): _pack("Ki — subtree walk", "ops (2, 10), (1, 100), (3, 1)", wframes, _sp(walked)),
    }


def _preorder(n, parents):
    ch = _children(n, parents)
    order = []

    def go(u):
        order.append(u)
        for v in ch[u]:
            go(v)

    go(1)
    pos = [0] * (n + 1)
    for i, u in enumerate(order):
        pos[u] = i
    sz = [1] * (n + 1)
    for u in reversed(order):
        for v in ch[u]:
            sz[u] += sz[v]
    return order, pos, sz


def _mil_forms():
    parents = [1, 1, 1, 3, 5, 3, 5, 7]
    queries = [(3, 1), (1, 5), (3, 4), (7, 3), (1, 8), (1, 9)]
    order, pos, sz = _preorder(9, parents)
    assert order == [1, 2, 3, 5, 6, 8, 7, 9, 4]
    ans = [-1 if k > sz[u] else order[pos[u] + k - 1] for u, k in queries]
    assert ans == [3, 6, 8, -1, 9, 4]
    frames = [row_step("one preorder", [("pre", order, None)], _sp(order))]
    for u, k in queries:
        got = -1 if k > sz[u] else order[pos[u] + k - 1]
        frames.append(line_step(
            f"({u}, {k})",
            [f"block length {sz[u]}", f"officer {got}"],
            f"pos {pos[u]} + {k} - 1",
        ))

    ch = _children(9, parents)
    slow = []
    sframes = []
    for u, k in queries:
        seen = []

        def go(x, k=k, seen=seen):
            if len(seen) >= k:
                return
            seen.append(x)
            for v in ch[x]:
                go(v)

        go(u)
        got = -1 if len(seen) < k else seen[k - 1]
        slow.append(got)
        sframes.append(line_step(f"walk ({u}, {k})", [f"listed {_sp(seen)}", f"answer {got}"]))
    assert slow == ans
    return {
        ("mil", "pre"): _pack("Military — one preorder", "9 officers", frames, _sp(ans)),
        ("mil", "slow"): _pack("Military — search each query", "9 officers", sframes, _sp(slow)),
    }


def _desc_tables(n, parents):
    ch = _children(n, parents)
    tin = [0] * (n + 1)
    tout = [0] * (n + 1)
    dep = [0] * (n + 1)
    at = [[] for _ in range(n)]
    clock = [0]

    def go(u):
        tin[u] = clock[0]
        at[dep[u]].append(clock[0])
        clock[0] += 1
        for v in ch[u]:
            dep[v] = dep[u] + 1
            go(v)
        tout[u] = clock[0] - 1

    go(1)
    return tin, tout, dep, at, ch


def _desc_answer(n, parents, queries):
    tin, tout, dep, at, ch = _desc_tables(n, parents)
    ans = []
    for u, d in queries:
        if d >= n or d < dep[u]:
            ans.append(0)
            continue
        row = at[d]
        lo = next((i for i, t in enumerate(row) if t >= tin[u]), len(row))
        hi = next((i for i, t in enumerate(row) if t > tout[u]), len(row))
        ans.append(hi - lo)
    return ans, tin, tout, dep, at, ch


def _desc_forms():
    parents = [1, 1, 2, 2, 4, 2]
    queries = [(1, 2), (7, 2), (4, 1), (5, 5)]
    ans, tin, tout, dep, at, ch = _desc_answer(7, parents, queries)
    assert ans == [3, 1, 0, 0]
    verts = []
    for v in range(1, 8):
        if dep[v] == 2:
            verts.append(v)
    assert verts == [4, 5, 7]
    frames = [line_step("depth 2", [f"vertices {_sp(verts)}", f"entries {_sp(at[2])}"], "filled during the walk")]
    for (u, d), got in zip(queries, ans):
        frames.append(line_step(
            f"({u}, {d})",
            [f"depth of {u} is {dep[u]}", f"window {tin[u]}..{tout[u]}", f"count {got}"],
        ))
    walk = []
    wframes = []
    for u, d in queries:
        total = 0

        def go(x, d=d):
            nonlocal total
            total += int(dep[x] == d)
            for v in ch[x]:
                go(v)

        go(u)
        walk.append(total)
        wframes.append(line_step(f"walk ({u}, {d})", [f"count {total}"]))
    assert walk == ans
    text = _sp(ans)
    return {
        ("desc", "bin"): _pack("Count descendants — binary search", "parents 1 1 2 2 4 2", frames, text),
        ("desc", "walk"): _pack("Count descendants — walk the subtree", "parents 1 1 2 2 4 2", wframes, _sp(walk)),
    }


S1 = ["#..#", "..#.", "#..."]


def _maze_orders(grid, finish=False):
    g = [list(row) for row in grid]
    n, m = len(g), len(g[0])
    start = next((r, c) for r in range(n) for c in range(m) if g[r][c] == ".")
    seen = [[False] * m for _ in range(n)]
    order = []

    def go(r, c):
        seen[r][c] = True
        if not finish:
            order.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < m and g[nr][nc] == "." and not seen[nr][nc]:
                go(nr, nc)
        if finish:
            order.append((r, c))

    go(*start)
    return g, order


def _paint(grid, cells):
    g = [list(row) for row in grid]
    for r, c in cells:
        g[r][c] = "X"
    return ["".join(row) for row in g]


def _maze_forms():
    g, disc = _maze_orders(S1, finish=False)
    walled = _paint(S1, disc[-2:])
    assert walled == ["#.X#", "X.#.", "#..."]
    _, fin = _maze_orders(S1, finish=True)
    finished = _paint(S1, fin[:2])
    assert finished == ["#..#", "..#X", "#..X"]
    base = [list(row) for row in S1]
    marks = {(r, c): "red" for r, c in disc[-2:]}
    dframes = [
        line_step("discovery order", [_sp(f"({r},{c})" for r, c in disc)], "keep the first s − k"),
        grid_step("wall the last two discovered", base, marks=marks),
        grid_step("six empties stay joined", [list(row) for row in walled],
                  marks={(r, c): "red" for r, row in enumerate(walled) for c, ch in enumerate(row) if ch == "X"}),
    ]
    fmarks = {(r, c): "red" for r, c in fin[:2]}
    fframes = [
        line_step("finishing order", [_sp(f"({r},{c})" for r, c in fin[:4]) + " …"], "wall the first k"),
        grid_step("wall the first two finished", base, marks=fmarks),
        grid_step("the other legal pair", [list(row) for row in finished],
                  marks={(r, c): "red" for r, row in enumerate(finished) for c, ch in enumerate(row) if ch == "X"}),
    ]
    return {
        ("maze", "disc"): _pack("Maze — discovery prefix", "k = 2", dframes, " | ".join(walled)),
        ("maze", "finish"): _pack("Maze — finishing prefix", "k = 2", fframes, " | ".join(finished)),
    }


def _path_forms():
    graph = [[1, 2], [3], [3], []]
    ans = []
    path = [0]
    frames = [line_step("start", ["0"])]

    def go(u):
        if u == len(graph) - 1:
            ans.append(path[:])
            frames.append(line_step("record", [_sp(path)], "copy the stack"))
            return
        for v in graph[u]:
            path.append(v)
            frames.append(line_step(f"push {v}", [_sp(path)]))
            go(v)
            path.pop()

    go(0)
    assert ans == [[0, 1, 3], [0, 2, 3]]
    text = "  ".join("[" + ", ".join(show(x) for x in p) + "]" for p in ans)
    return {("paths", "back"): _pack("All paths — backtracking", "0→1→3, 0→2→3", frames, text)}


def _gold_plain(grid):
    g = [row[:] for row in grid]
    R, C = len(g), len(g[0])
    frames = []
    best_seen = 0

    def go_sum(r, c, so_far):
        nonlocal best_seen
        val = g[r][c]
        g[r][c] = 0
        so_far += val
        if so_far > best_seen:
            best_seen = so_far
            frames.append(grid_step(f"sum {so_far}", [row[:] for row in g], eq=f"best {so_far}"))
        extra = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and g[nr][nc]:
                extra = max(extra, go_sum(nr, nc, so_far))
        g[r][c] = val
        return val + extra

    best = 0
    for r in range(R):
        for c in range(C):
            if g[r][c]:
                best = max(best, go_sum(r, c, 0))
    return best, frames


def _gold_prune(grid):
    g = [row[:] for row in grid]
    R, C = len(g), len(g[0])
    best = 0
    frames = []
    total = sum(sum(row) for row in g)
    noted_prune = False

    def go(r, c, got, left):
        nonlocal best, noted_prune
        val = g[r][c]
        got += val
        left -= val
        if got > best:
            best = got
            frames.append(grid_step(f"sum {got}", [row[:] for row in g], eq=f"best {got}"))
        if got + left <= best:
            if not noted_prune:
                noted_prune = True
                frames.append(line_step(
                    "prune",
                    [f"got {got}", f"left {left}", f"best {best}"],
                    "got + left cannot win",
                ))
            return
        g[r][c] = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and g[nr][nc]:
                go(nr, nc, got, left)
        g[r][c] = val

    for r in range(R):
        for c in range(C):
            if g[r][c]:
                go(r, c, 0, total)
    if not noted_prune:
        frames.append(line_step(
            "bound stays open",
            [f"board {show(total)}", f"best {show(best)}"],
            "got + gold still counted is the whole board",
        ))
    frames.append(line_step("answer", [show(best)], f"best {show(best)}"))
    return best, frames


def _gold_forms():
    grid = [[0, 6, 0], [5, 8, 7], [0, 9, 0]]
    best, frames = _gold_plain(grid)
    assert best == 24
    pbest, pframes = _gold_prune(grid)
    assert pbest == 24
    # Drop the duplicate grid shot that the prune records on the same sum.
    return {
        ("gold", "restore"): _pack("Maximum gold — restore on exit", "9 + 8 + 7", frames, best),
        ("gold", "prune"): _pack("Maximum gold — prune by gold left", "9 + 8 + 7", pframes, pbest),
    }


EDGES = [[0, 3], [0, 4], [1, 3], [2, 4], [2, 7], [3, 5], [3, 6], [3, 7], [4, 6]]


def _anc_rev(n, edges):
    rev = [[] for _ in range(n)]
    for u, v in edges:
        rev[v].append(u)
    ans = []
    frames = []
    for s in range(n):
        seen = [False] * n
        st = [s]
        while st:
            u = st.pop()
            for v in rev[u]:
                if not seen[v]:
                    seen[v] = True
                    st.append(v)
        row = [i for i in range(n) if seen[i]]
        ans.append(row)
        frames.append(line_step(f"vertex {s}", [_sp(row) if row else "none"], "index order"))
    return ans, frames


def _anc_union(n, edges):
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    sets = [set() for _ in range(n)]
    frames = []
    q = deque(i for i in range(n) if indeg[i] == 0)
    while q:
        u = q.popleft()
        frames.append(line_step(
            f"pop {u}",
            [f"ancestors {_sp(sorted(sets[u])) or 'none'}"],
            "predecessors are finished",
        ))
        for v in adj[u]:
            sets[v].add(u)
            sets[v] |= sets[u]
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    ans = [sorted(s) for s in sets]
    return ans, frames


def _fmt_anc(rows):
    parts = []
    for i, row in enumerate(rows):
        parts.append(f"{i}: " + (_sp(row) if row else "—"))
    return "   ".join(parts)


def _anc_forms():
    rev, rframes = _anc_rev(8, EDGES)
    fwd, fframes = _anc_union(8, EDGES)
    assert rev == fwd == [
        [], [], [], [0, 1], [0, 2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3],
    ]
    text = _fmt_anc(rev)
    return {
        ("anc", "rev"): _pack("Ancestors — reversed search", "8 vertices", rframes, text),
        ("anc", "union"): _pack("Ancestors — unions along the arcs", "8 vertices", fframes, text),
    }


def _solve(grid):
    g = [list(row) for row in grid]
    n, m = len(g), len(g[0])
    goods = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    sealed = []
    for r in range(n):
        for c in range(m):
            if g[r][c] != "B":
                if g[r][c] == "G":
                    goods.append((r, c))
                continue
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n and 0 <= nc < m):
                    continue
                if g[nr][nc] == "G":
                    return False, g, sealed, goods
                if g[nr][nc] == ".":
                    g[nr][nc] = "#"
                    sealed.append((nr, nc))
    if not goods:
        return True, g, sealed, goods
    er, ec = n - 1, m - 1
    if g[er][ec] == "#":
        return False, g, sealed, goods
    seen = [[False] * m for _ in range(n)]
    seen[er][ec] = True
    q = deque([(er, ec)])
    reached = 0
    while q:
        r, c = q.popleft()
        if g[r][c] == "G":
            reached += 1
        if g[r][c] == "B":
            return False, g, sealed, goods
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < m and not seen[nr][nc] and g[nr][nc] != "#":
                seen[nr][nc] = True
                q.append((nr, nc))
    return reached == len(goods), g, sealed, goods


def _solve_forms():
    grid = ["#B.", "#..", "GG."]
    ok, g, sealed, goods = _solve(grid)
    assert ok is True and len(goods) == 2
    assert _solve(["#B", "B."])[0] is True
    assert _solve(["#B", "G."])[0] is False
    base = [list(row) for row in grid]
    frames = [
        grid_step("before the seal", base, marks={(0, 1): "red", (2, 0): "green", (2, 1): "green", (2, 2): "amber"}),
        grid_step("empty neighbours of B become walls", [row[:] for row in g],
                  marks={(r, c): "red" for r, c in sealed}),
        line_step("search from the exit", [f"goods reached {len(goods)}", "bad person not reached"], "answer yes"),
    ]
    return {("solve", "seal"): _pack("Solve the maze — seal, then search", "#B. / #.. / GG.", frames, ok)}


_CACHE = None


def get(problem, form):
    """Return ``(title, input line, steps, final)`` for one solution form."""
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        for part in (
            _trip_forms(), _trip2_forms(), _wb_forms(), _score_forms(), _ki_forms(),
            _mil_forms(), _desc_forms(), _maze_forms(), _path_forms(), _gold_forms(),
            _anc_forms(), _solve_forms(),
        ):
            _CACHE.update(part)
    return _CACHE[(problem, form)]
