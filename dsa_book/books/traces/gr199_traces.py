"""Step frames for the graphs introduction.

Drawing lives in ``dsa_viz.story``. ``get`` returns data only.
Each story is the chapter sample, at most eight frames.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import chip_step, grid_step, line_step, row_step, run_call, show


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    if not 1 <= len(steps) <= 8:
        raise AssertionError(f"{title}: {len(steps)} frames")
    return (title, inp, steps, text)


def _judge_score():
    n, trust = 3, [(1, 3), (2, 3)]
    score = [0, 0, 0]
    steps = [row_step("scores start at 0", [("score", score[:], None)], "n = 3")]
    for a, b in trust:
        score[a - 1] -= 1
        score[b - 1] += 1
        steps.append(row_step(
            f"{a} → {b}",
            [("score", score[:], b - 1)],
            f"score[{b}] = {score[b - 1]}",
        ))
    found = next(i for i in range(1, 4) if score[i - 1] == n - 1)
    steps.append(row_step(
        f"score[{found}] = {n - 1}",
        [("score", score[:], found - 1)],
        f"{score[found - 1]} = {n - 1}",
    ))
    return _pack("Town judge — score", "n = 3, trust = [[1, 3], [2, 3]]", steps, found)


def _judge_elim():
    n = 3
    arc = [[False] * (n + 1) for _ in range(n + 1)]
    for a, b in ((1, 3), (2, 3)):
        arc[a][b] = True
    cand = 1
    steps = [chip_step("candidate 1", [("cand", cand, "indigo")], "nobody eliminated yet")]
    for v in range(2, n + 1):
        if arc[cand][v]:
            old = cand
            cand = v
            steps.append(chip_step(
                f"{old} trusts {v}",
                [("cand", cand, "amber")],
                f"cand = {cand}",
            ))
        else:
            steps.append(chip_step(
                f"no arc {cand} → {v}",
                [("cand", cand, "indigo")],
                f"cand stays {cand}",
            ))
    ok = all(not arc[cand][v] and arc[v][cand] for v in range(1, n + 1) if v != cand)
    steps.append(chip_step(
        "verify in-degree and out-degree",
        [("cand", cand, "green" if ok else "red")],
        f"judge = {cand}" if ok else "reject",
    ))
    return _pack("Town judge — elimination", "n = 3, trust = [[1, 3], [2, 3]]", steps, cand if ok else -1)


def _rank_pairs():
    n, roads = 4, [(0, 1), (0, 3), (1, 2), (1, 3)]
    deg = [0] * n
    edge = [[False] * n for _ in range(n)]
    for u, v in roads:
        deg[u] += 1
        deg[v] += 1
        edge[u][v] = edge[v][u] = True
    steps = [row_step("degrees", [("deg", deg[:], None)], "2, 3, 1, 2")]
    best = 0
    for u, v in ((a, b) for a in range(n) for b in range(a + 1, n)):
        share = 1 if edge[u][v] else 0
        rank = deg[u] + deg[v] - share
        best = max(best, rank)
        steps.append(chip_step(
            f"pair {u}, {v}",
            [("rank", rank, "green" if rank == best else "indigo"), ("best", best, "amber")],
            f"{deg[u]} + {deg[v]} − {share} = {rank}",
        ))
    return _pack("Network rank — all pairs", "n = 4, roads = [[0, 1], [0, 3], [1, 2], [1, 3]]", steps, best)


def _rank_top():
    n, roads = 4, [(0, 1), (0, 3), (1, 2), (1, 3)]
    deg = [0] * n
    adj = [set() for _ in range(n)]
    for u, v in roads:
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
    steps = [row_step("degrees", [("deg", deg[:], deg.index(max(deg)))], f"top = {deg.index(max(deg))}")]
    tops = [i for i in range(n) if deg[i] == max(deg)]
    if len(tops) > 1:
        raise AssertionError("sample has a unique maximum")
    u = tops[0]
    best = 0
    for v in range(n):
        if v == u:
            continue
        share = 1 if v in adj[u] else 0
        rank = deg[u] + deg[v] - share
        best = max(best, rank)
        steps.append(chip_step(
            f"{u} with {v}",
            [("rank", rank, "green" if rank == best else "indigo"), ("best", best, "amber")],
            f"{deg[u]} + {deg[v]} − {share} = {rank}",
        ))
    return _pack("Network rank — highest degrees", "n = 4, roads = [[0, 1], [0, 3], [1, 2], [1, 3]]", steps, best)


def _path_bfs():
    edges = [(0, 1), (1, 2), (2, 0)]
    adj = [[] for _ in range(3)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = [False] * 3
    seen[0] = True
    q = deque([0])
    steps = [line_step("mark 0", ["queue = [0]"], "source is discovered")]
    answer = False
    while q and not answer:
        u = q.popleft()
        for v in adj[u]:
            if seen[v]:
                continue
            if v == 2:
                steps.append(line_step(
                    f"{u} sees {v}",
                    [f"queue was {[u] + list(q)}", "2 is the destination"],
                    "return true",
                ))
                answer = True
                break
            seen[v] = True
            q.append(v)
            steps.append(line_step(
                f"push {v}",
                [f"from {u}", f"queue = {list(q)}"],
                f"mark {v}",
            ))
    return _pack("Path exists — breadth-first search", "n = 3, edges = [[0, 1], [1, 2], [2, 0]], 0 → 2", steps, answer)


def _path_dsu():
    parent = [0, 1, 2]
    steps = [row_step("each vertex is its own root", [("parent", parent[:], None)], "")]

    def find(x, snap=False):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in ((0, 1), (1, 2), (2, 0)):
        a, b = find(u), find(v)
        if a != b:
            parent[a] = b
        steps.append(row_step(
            f"union {u}, {v}",
            [("parent", parent[:], b if a != b else None)],
            "same root" if a == b else f"parent[{a}] = {b}",
        ))
    same = find(0) == find(2)
    steps.append(chip_step(
        "compare roots",
        [("root 0", find(0), "green"), ("root 2", find(2), "green")],
        "same component" if same else "different",
    ))
    return _pack("Path exists — disjoint sets", "n = 3, edges = [[0, 1], [1, 2], [2, 0]], 0 → 2", steps, same)


def _keys_stack():
    rooms = [[1], [2], [3], []]
    seen = [False] * 4
    seen[0] = True
    st = [0]
    opened = 1
    steps = [chip_step("room 0 is open", [("opened", opened, "indigo"), ("stack", "0", "amber")], "count the start")]
    while st:
        u = st.pop()
        for v in rooms[u]:
            if not seen[v]:
                seen[v] = True
                opened += 1
                st.append(v)
                steps.append(chip_step(
                    f"open {v} from {u}",
                    [("opened", opened, "indigo"), ("stack", " ".join(map(str, st)) or "—", "amber")],
                    f"opened = {opened}",
                ))
    return _pack("Keys and rooms — stack", "[[1], [2], [3], []]", steps, opened == 4)


def _keys_rec():
    rooms = [[1], [2], [3], []]

    def build(log):
        seen = [False] * 4

        def go(u):
            log.call(f"go({u})")
            seen[u] = True
            total = 1
            children = False
            for v in rooms[u]:
                if not seen[v]:
                    total += go(v)
                    children = True
            if not children:
                return log.base(total, f"go({u}) = {total}")
            return log.ret(total, f"go({u}) = {total}")

        return go(0) == 4

    ans, steps = run_call(build, limit=8)
    return _pack("Keys and rooms — recursion", "[[1], [2], [3], []]", steps, ans)


def _plan_grid(g, colors):
    marks = {}
    view = []
    for r, row in enumerate(g):
        view.append([])
        for c, ch in enumerate(row):
            if (r, c) in colors:
                view[-1].append(colors[(r, c)])
                marks[(r, c)] = "green" if colors[(r, c)] != "#" else "red"
            elif ch == "#":
                view[-1].append("#")
                marks[(r, c)] = "red"
            else:
                view[-1].append(".")
    return view, marks


PLAN = ["########", "#..#...#", "####.#.#", "#..#...#", "########"]


def _rooms_flood():
    g = [list(row) for row in PLAN]
    n, m = len(g), len(g[0])
    names = "ABC"
    colors = {}
    steps = []
    rooms = 0
    for r in range(n):
        for c in range(m):
            if g[r][c] != ".":
                continue
            rooms += 1
            name = names[rooms - 1]
            q = deque([(r, c)])
            g[r][c] = "#"
            colors[(r, c)] = name
            while q:
                rr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < n and 0 <= nc < m and g[nr][nc] == ".":
                        g[nr][nc] = "#"
                        colors[(nr, nc)] = name
                        q.append((nr, nc))
            view, marks = _plan_grid([list(row) for row in PLAN], colors)
            steps.append(grid_step(
                f"room {name} from ({r}, {c})",
                view,
                hi=(r, c),
                marks=marks,
                eq=f"rooms = {rooms}",
            ))
    return _pack("Counting rooms — flood fill", "5 × 8 plan", steps, rooms)


def _rooms_dsu():
    grid = PLAN
    n, m = len(grid), len(grid[0])
    parent = list(range(n * m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unite(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    snaps = []
    for r in range(n):
        for c in range(m):
            if grid[r][c] != ".":
                continue
            i = r * m + c
            if r and grid[r - 1][c] == ".":
                unite(i, (r - 1) * m + c)
            if c and grid[r][c - 1] == ".":
                unite(i, r * m + (c - 1))
        if any(ch == "." for ch in grid[r]):
            snaps.append(r)
    letters = {}

    def paint(upto):
        colors = {}
        for r in range(upto + 1):
            for c in range(m):
                if grid[r][c] != ".":
                    continue
                root = find(r * m + c)
                if root not in letters:
                    letters[root] = "ABC"[len(letters)]
                colors[(r, c)] = letters[root]
        return colors

    steps = []
    for r in snaps:
        colors = paint(r)
        view, marks = _plan_grid([list(row) for row in grid], colors)
        roots = {find(rr * m + cc) for rr in range(r + 1) for cc in range(m) if grid[rr][cc] == "."}
        steps.append(grid_step(
            f"after row {r}",
            view,
            marks=marks,
            eq=f"{len(roots)} root{'s' if len(roots) != 1 else ''} so far",
        ))
    final = len({find(r * m + c) for r in range(n) for c in range(m) if grid[r][c] == "."})
    return _pack("Counting rooms — disjoint sets", "5 × 8 plan", steps, final)


ISLAND = [
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
]


def _island_areas(sink):
    g = [row[:] for row in ISLAND]
    R, C = len(g), len(g[0])
    events = []
    best = 0
    for sr in range(R):
        for sc in range(C):
            if g[sr][sc] != 1:
                continue
            area = sink(g, sr, sc, R, C)
            best = max(best, area)
            events.append((sr, sc, area, best))
    return events, best


def _sink_cell(g, r, c, R, C):
    if r < 0 or c < 0 or r >= R or c >= C or g[r][c] == 0:
        return 0
    g[r][c] = 0
    return 1 + _sink_cell(g, r + 1, c, R, C) + _sink_cell(g, r - 1, c, R, C) + _sink_cell(g, r, c + 1, R, C) + _sink_cell(g, r, c - 1, R, C)


def _area_sink():
    events, best = _island_areas(_sink_cell)
    steps = [
        chip_step(
            f"sink ({sr}, {sc})",
            [("area", area, "green" if area == best else "indigo"), ("best", cur, "amber")],
            f"1 + four neighbours = {area}",
        )
        for sr, sc, area, cur in events
    ]
    return _pack("Max area — sink", "official island grid", steps, best)


def _bfs_cell(g, sr, sc, R, C):
    q = deque([(sr, sc)])
    g[sr][sc] = 0
    area = 0
    trace = []
    while q:
        r, c = q.popleft()
        area += 1
        trace.append((r, c, area))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and g[nr][nc] == 1:
                g[nr][nc] = 0
                q.append((nr, nc))
    return area, trace


def _area_queue():
    g = [row[:] for row in ISLAND]
    R, C = len(g), len(g[0])
    best = 0
    winner = None
    for sr in range(R):
        for sc in range(C):
            if g[sr][sc] != 1:
                continue
            area, trace = _bfs_cell(g, sr, sc, R, C)
            if area >= best:
                best = area
                winner = trace
    # Show the island that sets the maximum: first pop, a middle pop, and the last.
    picks = [0, len(winner) // 2, len(winner) - 1]
    steps = []
    for i in picks:
        r, c, area = winner[i]
        steps.append(chip_step(
            f"pop ({r}, {c})",
            [("area", area, "green" if i == len(winner) - 1 else "indigo")],
            "mark was already written on the push",
        ))
    steps.append(chip_step(
        "island closed",
        [("best", best, "green")],
        f"best = {best}",
    ))
    return _pack("Max area — queue", "official island grid", steps, best)


def _buttons_bfs():
    n, m = 4, 6
    hi = 2 * max(n, m)
    dist = [-1] * (hi + 1)
    dist[n] = 0
    q = deque([n])
    steps = [chip_step("start", [("x", n, "indigo"), ("dist", 0, "amber")], f"window 1..{hi}")]
    answer = None
    while q and answer is None:
        x = q.popleft()
        for y in (2 * x, x - 1):
            if y < 1 or y > hi or dist[y] != -1:
                continue
            dist[y] = dist[x] + 1
            steps.append(chip_step(
                f"{x} → {y}",
                [("y", y, "green" if y == m else "indigo"), ("dist", dist[y], "amber")],
                f"dist[{y}] = {dist[y]}",
            ))
            if y == m:
                answer = dist[y]
                break
            q.append(y)
            if len(steps) >= 7:
                break
        if len(steps) >= 7 and answer is None:
            break
    if answer is None:
        raise AssertionError("two buttons sample did not finish inside the frame budget")
    return _pack("Two buttons — breadth-first search", "n = 4, m = 6", steps, answer)


def _buttons_rev():
    n, m = 4, 6
    presses = 0
    steps = [chip_step("target above n", [("m", m, "indigo"), ("presses", presses, "amber")], "6 > 4")]
    while m > n:
        if m % 2:
            steps.append(chip_step("odd, undo a decrement", [("m", m + 1, "indigo")], f"{m} + 1"))
            m += 1
        else:
            steps.append(chip_step("even, undo a doubling", [("m", m // 2, "indigo")], f"{m} / 2"))
            m //= 2
        presses += 1
    answer = presses + (n - m)
    steps.append(chip_step(
        "add the blue tail",
        [("m", m, "amber"), ("presses", answer, "green")],
        f"{presses} + ({n} − {m}) = {answer}",
    ))
    return _pack("Two buttons — reverse walk", "n = 4, m = 6", steps, answer)


def _lock_code(x):
    return f"{x:04d}"


def _lock_neighbours(x):
    P = (1000, 100, 10, 1)
    out = []
    for w in range(4):
        g = x // P[w] % 10
        for step in (1, 9):
            ng = (g + step) % 10
            out.append(x + (ng - g) * P[w])
    return out


def _lock_bfs():
    dead = {int(s) for s in ("0201", "0101", "0102", "1212", "2002")}
    tgt = 202
    dist = [-1] * 10000
    parent = [-1] * 10000
    dist[0] = 0
    q = deque([0])
    found = False
    while q and not found:
        x = q.popleft()
        for y in _lock_neighbours(x):
            if y in dead or dist[y] != -1:
                continue
            dist[y] = dist[x] + 1
            parent[y] = x
            if y == tgt:
                found = True
                break
            q.append(y)
    path = []
    y = tgt
    while y != -1:
        path.append(y)
        y = parent[y]
    path.reverse()
    steps = []
    for i, code in enumerate(path):
        steps.append(chip_step(
            f"distance {i}",
            [("code", _lock_code(code), "green" if code == tgt else "indigo"), ("dist", i, "amber")],
            "target" if code == tgt else "first time this code is reached",
        ))
    return _pack("Open the lock — breadth-first search", 'deadends block 0202, target 0202', steps, dist[tgt])


def _lock_bi():
    dead = {int(s) for s in ("0201", "0101", "0102", "1212", "2002")}
    tgt = 202
    fa = {0: 0}
    fb = {tgt: 0}
    qa, qb = deque([0]), deque([tgt])
    steps = [line_step(
        "two fronts",
        ["start 0000 at 0", "target 0202 at 0"],
        "expand the smaller front",
    )]
    answer = None
    guard = 0
    while qa and qb and answer is None and guard < 30:
        guard += 1
        side = "start"
        if len(qa) > len(qb):
            qa, qb = qb, qa
            fa, fb = fb, fa
            side = "target"
        layer = len(qa)
        hit = None
        for _ in range(layer):
            x = qa.popleft()
            for y in _lock_neighbours(x):
                if y in dead or y in fa:
                    continue
                if y in fb:
                    hit = (x, y, fa[x] + 1 + fb[y])
                    break
                fa[y] = fa[x] + 1
                qa.append(y)
            if hit:
                break
        if hit:
            x, y, answer = hit
            steps.append(line_step(
                f"{side} side meets",
                [f"{_lock_code(x)} touches {_lock_code(y)}", f"distance {answer}"],
                f"{fa[x] if x in fa else '?'} + 1 + other",
            ))
        elif len(steps) < 7:
            steps.append(line_step(
                f"expand {side}",
                [f"layer of {layer}", f"front now {len(qa)}"],
                "whole layer, one distance",
            ))
    # The meeting uses fa[x] after a possible swap, so recompute the displayed sum from the answer.
    if steps and answer is not None and "meets" in steps[-1]["title"]:
        steps[-1] = line_step(
            steps[-1]["title"],
            steps[-1]["lines"],
            f"distance = {answer}",
        )
    return _pack("Open the lock — two frontiers", 'deadends block 0202, target 0202', steps, answer)


def _badge_sim():
    p = [0, 2, 3, 2]
    n = 3
    ans = []
    steps = []
    for s in range(1, n + 1):
        seen = [False] * (n + 1)
        walk = []
        v = s
        while not seen[v]:
            seen[v] = True
            walk.append(v)
            v = p[v]
        ans.append(v)
        steps.append(line_step(
            f"start {s}",
            [" → ".join(map(str, walk + [v])), f"first repeat is {v}"],
            f"ans[{s}] = {v}",
        ))
    return _pack("Badge — resimulate", "p = [2, 3, 2]", steps, ans)


def _badge_peel():
    p = [0, 2, 3, 2]
    n = 3
    indeg = [0] * (n + 1)
    for i in range(1, n + 1):
        indeg[p[i]] += 1
    q = deque(i for i in range(1, n + 1) if indeg[i] == 0)
    on = [True] * (n + 1)
    steps = [row_step("in-degrees", [("in", indeg[1:], None)], "peel in-degree 0")]
    while q:
        u = q.popleft()
        on[u] = False
        indeg[p[u]] -= 1
        if indeg[p[u]] == 0:
            q.append(p[u])
        steps.append(chip_step(
            f"peel {u}",
            [("next", p[u], "indigo"), ("in", indeg[p[u]], "amber")],
            f"in[{p[u]}] = {indeg[p[u]]}",
        ))
    ans = [0] * (n + 1)
    for i in range(1, n + 1):
        if on[i]:
            ans[i] = i
    steps.append(line_step(
        "cycle vertices answer themselves",
        [f"ans[{i}] = {i}" for i in range(1, n + 1) if on[i]],
        "2 and 3 remain",
    ))
    for i in range(1, n + 1):
        if ans[i]:
            continue
        path = []
        v = i
        while not ans[v]:
            path.append(v)
            v = p[v]
        for u in path:
            ans[u] = ans[v]
        steps.append(line_step(
            f"tail {path[0]} copies {v}",
            [" → ".join(map(str, path + [v]))],
            f"ans = {ans[1:]}",
        ))
    return _pack("Badge — peel, then copy", "p = [2, 3, 2]", steps, ans[1:])


def _cycle_time():
    edges = [3, 3, 4, 2, 3]
    n = len(edges)
    when = [-1] * n
    time = 0
    best = -1
    steps = []
    for s in range(n):
        if when[s] != -1:
            continue
        start = time
        v = s
        walk = []
        while v != -1 and when[v] == -1:
            when[v] = time
            walk.append(f"{v}@{time}")
            time += 1
            v = edges[v]
        length = None
        if v != -1 and when[v] >= start:
            length = time - when[v]
            best = max(best, length)
        title = f"walk from {s}"
        lines = [" ".join(walk)]
        if v == -1:
            lines.append("falls off")
        elif when[v] >= start:
            lines.append(f"back to {v}, length {length}")
        else:
            lines.append(f"hit {v}, stamped before this walk")
        steps.append(line_step(title, lines, f"best = {best}"))
    return _pack("Longest cycle — timestamps", "edges = [3, 3, 4, 2, 3]", steps, best)


def _cycle_peel():
    edges = [3, 3, 4, 2, 3]
    n = len(edges)
    indeg = [0] * n
    for v in edges:
        if v != -1:
            indeg[v] += 1
    q = deque(i for i in range(n) if indeg[i] == 0)
    alive = [True] * n
    steps = [row_step("in-degrees", [("in", indeg[:], None)], "0 and 1 have none")]
    peeled = []
    while q:
        u = q.popleft()
        alive[u] = False
        peeled.append(u)
        v = edges[u]
        if v != -1:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    steps.append(line_step("peeled", [", ".join(map(str, peeled))], "not on a cycle"))
    best = -1
    seen = [False] * n
    for s in range(n):
        if not alive[s] or seen[s]:
            continue
        length = 0
        walk = []
        v = s
        while not seen[v]:
            seen[v] = True
            length += 1
            walk.append(str(v))
            v = edges[v]
        best = max(best, length)
        steps.append(line_step(
            f"cycle from {s}",
            [" → ".join(walk)],
            f"length {length}",
        ))
    steps.append(chip_step("longest survivor", [("best", best, "green")], f"best = {best}"))
    return _pack("Longest cycle — peel", "edges = [3, 3, 4, 2, 3]", steps, best)


def _dup_floyd():
    nums = [1, 3, 4, 2, 2]
    t, h = nums[0], nums[nums[0]]
    steps = []
    while True:
        steps.append(chip_step(
            "phase 1" if t != h else "phase 1 meets",
            [("tortoise", t, "cyan"), ("hare", h, "amber" if t != h else "green")],
            "one step, two steps" if t != h else "both on the cycle",
        ))
        if t == h:
            break
        t, h = nums[t], nums[nums[h]]
    t = 0
    while True:
        steps.append(chip_step(
            "phase 2 restarts at 0" if len(steps) == 4 else ("phase 2 meets" if t == h else "phase 2"),
            [("tortoise", t, "cyan" if t != h else "green"), ("hare", h, "amber" if t != h else "green")],
            "tortoise back at 0" if t == 0 and h != 0 else ("entrance" if t == h else "both take one step"),
        ))
        if t == h:
            break
        t, h = nums[t], nums[h]
    return _pack("Duplicate — tortoise and hare", "nums = [1, 3, 4, 2, 2]", steps, t)


def _dup_count():
    nums = [1, 3, 4, 2, 2]
    lo, hi = 1, len(nums) - 1
    steps = []
    while lo < hi:
        mid = (lo + hi) // 2
        cnt = sum(x <= mid for x in nums)
        if cnt > mid:
            hi = mid
            way = "duplicate ≤ mid"
        else:
            lo = mid + 1
            way = "duplicate > mid"
        steps.append(chip_step(
            f"mid {mid}",
            [("count", cnt, "indigo"), ("lo", lo, "amber"), ("hi", hi, "amber")],
            f"{cnt} values ≤ {mid}: {way}",
        ))
    steps.append(chip_step("range collapsed", [("value", lo, "green")], f"lo = hi = {lo}"))
    return _pack("Duplicate — counting", "nums = [1, 3, 4, 2, 2]", steps, lo)


def _dup_brent():
    nums = [1, 3, 4, 2, 2]
    tortoise, hare = 0, nums[0]
    power = lam = 1
    steps = [chip_step(
        "hare one step ahead",
        [("tortoise", tortoise, "cyan"), ("hare", hare, "amber"), ("λ", lam, "violet")],
        "power = 1",
    )]
    while tortoise != hare:
        if power == lam:
            tortoise = hare
            power *= 2
            lam = 0
            steps.append(chip_step(
                "tortoise jumps to the hare",
                [("tortoise", tortoise, "cyan"), ("power", power, "violet")],
                f"power = {power}",
            ))
        hare = nums[hare]
        lam += 1
        if tortoise == hare or lam in (1, 2):
            steps.append(chip_step(
                "meet" if tortoise == hare else "hare steps",
                [("tortoise", tortoise, "green" if tortoise == hare else "cyan"),
                 ("hare", hare, "green" if tortoise == hare else "amber"),
                 ("λ", lam, "violet")],
                f"λ = {lam}" if tortoise == hare else "one step",
            ))
    tortoise = hare = 0
    for _ in range(lam):
        hare = nums[hare]
    steps.append(chip_step(
        f"hare is λ = {lam} ahead of 0",
        [("tortoise", 0, "cyan"), ("hare", hare, "amber")],
        "both will take one step",
    ))
    while tortoise != hare:
        tortoise, hare = nums[tortoise], nums[hare]
    steps.append(chip_step(
        "same entrance",
        [("tortoise", tortoise, "green"), ("hare", hare, "green")],
        f"duplicate = {tortoise}",
    ))
    if len(steps) > 8:
        steps = [steps[0], steps[1], steps[2], steps[-3], steps[-2], steps[-1]]
    return _pack("Duplicate — Brent", "nums = [1, 3, 4, 2, 2]", steps, tortoise)


_ALL = None


def get(problem, form):
    """Return ``(title, input line, steps, final)`` for one solution form."""
    global _ALL
    if _ALL is None:
        _ALL = {
            ("judge", "score"): _judge_score(),
            ("judge", "elim"): _judge_elim(),
            ("rank", "pairs"): _rank_pairs(),
            ("rank", "top"): _rank_top(),
            ("path", "bfs"): _path_bfs(),
            ("path", "dsu"): _path_dsu(),
            ("keys", "stack"): _keys_stack(),
            ("keys", "rec"): _keys_rec(),
            ("rooms", "flood"): _rooms_flood(),
            ("rooms", "dsu"): _rooms_dsu(),
            ("area", "sink"): _area_sink(),
            ("area", "queue"): _area_queue(),
            ("buttons", "bfs"): _buttons_bfs(),
            ("buttons", "rev"): _buttons_rev(),
            ("lock", "bfs"): _lock_bfs(),
            ("lock", "bi"): _lock_bi(),
            ("badge", "sim"): _badge_sim(),
            ("badge", "peel"): _badge_peel(),
            ("cycle", "time"): _cycle_time(),
            ("cycle", "peel"): _cycle_peel(),
            ("dup", "floyd"): _dup_floyd(),
            ("dup", "count"): _dup_count(),
            ("dup", "brent"): _dup_brent(),
        }
    return _ALL[(problem, form)]
