"""Step boards for every solution form in the trees introduction chapter.

Drawing lives in ``dsa_viz.story``. ``get`` returns data only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import chip_step, line_step, row_step, run_call, show

_CACHE = None


class _N:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def _level(vals):
    if not vals or vals[0] is None:
        return None
    root = _N(vals[0])
    q = [root]
    i = 1
    h = 0
    while h < len(q) and i < len(vals):
        u = q[h]
        h += 1
        if i < len(vals) and vals[i] is not None:
            u.left = _N(vals[i])
            q.append(u.left)
        i += 1
        if i < len(vals) and vals[i] is not None:
            u.right = _N(vals[i])
            q.append(u.right)
        i += 1
    return root


def _to_level(root):
    if root is None:
        return []
    q = [root]
    out = []
    h = 0
    while h < len(q):
        u = q[h]
        h += 1
        if u is None:
            out.append(None)
            continue
        out.append(u.val)
        q.append(u.left)
        q.append(u.right)
    while out and out[-1] is None:
        out.pop()
    return out


def _pack(title, inp, steps, answer):
    steps = list(steps)
    if len(steps) > 8:
        raise AssertionError(f"{title}: {len(steps)} frames")
    text = answer if isinstance(answer, str) else show(answer)
    return (title, inp, steps, text)


def _keep(frames, limit=8):
    if len(frames) <= limit:
        return frames
    last = len(frames) - 1
    mid = [round(i * last / (limit - 1)) for i in range(limit)]
    return [frames[i] for i in sorted(set(mid))]


def _count_walk():
    root = _level([1, 2, 3, 4, 5, 6])

    def build(log):
        def walk(u, name):
            log.call(name)
            if u is None:
                return log.base(0, f"{name} = 0")
            a = walk(u.left, f"{u.val}L")
            b = walk(u.right, f"{u.val}R")
            return log.ret(1 + a + b, f"{name} = {show(1 + a + b)}")

        return walk(root, "1")

    ans, steps = run_call(build)
    return _pack("Count — visit every node", "[1, 2, 3, 4, 5, 6]", steps, ans)


def _count_perfect():
    root = _level([1, 2, 3, 4, 5, 6])

    def build(log):
        def walk(u, name):
            log.call(name)
            if u is None:
                return log.base(0, f"{name} = 0")
            hl = hr = 0
            p = u
            while p is not None:
                hl += 1
                p = p.left
            p = u
            while p is not None:
                hr += 1
                p = p.right
            if hl == hr:
                return log.ret((1 << hl) - 1, f"{name} perfect, {hl} levels")
            a = walk(u.left, f"{u.val}L")
            b = walk(u.right, f"{u.val}R")
            return log.ret(1 + a + b, f"{name} = {show(1 + a + b)}")

        return walk(root, "1")

    ans, steps = run_call(build)
    return _pack("Count — perfect subtree", "[1, 2, 3, 4, 5, 6]", steps, ans)


def _count_bisect():
    root = _level([1, 2, 3, 4, 5, 6])
    h = 0
    p = root
    while p is not None:
        h += 1
        p = p.left

    def exists(idx):
        node = root
        for bit in range(h - 2, -1, -1):
            node = node.right if (idx >> bit) & 1 else node.left
        return node is not None

    lo, hi = 0, (1 << (h - 1)) - 1
    steps = [chip_step(
        f"h = {h}",
        [("lo", lo, "amber"), ("hi", hi, "indigo")],
        "last level has 4 slots",
    )]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        hit = exists(mid)
        if hit:
            lo = mid
        else:
            hi = mid - 1
        steps.append(chip_step(
            f"slot {mid} {'occupied' if hit else 'empty'}",
            [("lo", lo, "green" if lo == hi else "amber"), ("hi", hi, "indigo")],
            f"exists({mid}) = {show(hit)}",
        ))
    ans = (1 << (h - 1)) - 1 + lo + 1
    steps.append(chip_step(
        "full levels + last prefix",
        [("answer", ans, "green")],
        f"{show((1 << (h - 1)) - 1)} + {show(lo)} + 1",
    ))
    return _pack("Count — last level", "[1, 2, 3, 4, 5, 6]", steps, ans)


def _min_rec():
    root = _level([3, 9, 20, None, None, 15, 7])

    def build(log):
        def walk(u, name):
            log.call(name)
            if u is None:
                return log.base(0, f"{name} = 0")
            if u.left is None:
                r = walk(u.right, f"{u.val}R")
                return log.ret(1 + r, f"{name} = 1 + {show(r)}")
            if u.right is None:
                left = walk(u.left, f"{u.val}L")
                return log.ret(1 + left, f"{name} = 1 + {show(left)}")
            a = walk(u.left, f"{u.val}L")
            b = walk(u.right, f"{u.val}R")
            best = 1 + min(a, b)
            return log.ret(best, f"{name} = 1 + min({show(a)}, {show(b)})")

        return walk(root, "3")

    ans, steps = run_call(build)
    return _pack("Minimum depth — recursion", "[3, 9, 20, null, null, 15, 7]", steps, ans)


def _min_bfs():
    root = _level([3, 9, 20, None, None, 15, 7])
    q = [(root, 1)]
    steps = []
    ans = 0
    h = 0
    while h < len(q):
        u, d = q[h]
        h += 1
        leaf = u.left is None and u.right is None
        steps.append(line_step(
            f"pop {u.val}",
            [f"depth {d}", "leaf" if leaf else "enqueue the children"],
            f"{'return' if leaf else 'continue'} {d}",
        ))
        if leaf:
            ans = d
            break
        if u.left:
            q.append((u.left, d + 1))
        if u.right:
            q.append((u.right, d + 1))
    return _pack("Minimum depth — first leaf", "[3, 9, 20, null, null, 15, 7]", steps, ans)


def _same_rec():
    p = _level([1, 2, 3])
    q = _level([1, 2])

    def build(log):
        def walk(a, b, name):
            log.call(name)
            if a is None or b is None:
                return log.base(a is b, f"{name} = {show(a is b)}")
            if a.val != b.val:
                return log.base(False, f"{name} values differ")
            left = walk(a.left, b.left, name + "L")
            if not left:
                return log.ret(False, f"{name} = False")
            right = walk(a.right, b.right, name + "R")
            return log.ret(bool(left and right), f"{name} = {show(bool(left and right))}")

        return walk(p, q, "1,1")

    ans, steps = run_call(build)
    return _pack("Same tree — recursion", "p = [1, 2, 3], q = [1, 2]", steps, ans)


def _same_bfs():
    p = _level([1, 2, 3])
    q = _level([1, 2])
    qy = [(p, q)]
    steps = []
    ans = True
    h = 0
    while h < len(qy):
        a, b = qy[h]
        h += 1
        an = "·" if a is None else str(a.val)
        bn = "·" if b is None else str(b.val)
        if a is None or b is None:
            ans = a is b
            steps.append(line_step(f"pair {an}, {bn}", ["one side is missing"], f"same = {show(ans)}"))
            if not ans:
                break
            continue
        if a.val != b.val:
            ans = False
            steps.append(line_step(f"pair {an}, {bn}", ["values differ"], "same = False"))
            break
        steps.append(line_step(f"pair {an}, {bn}", ["values match", "enqueue both children"], "continue"))
        qy.append((a.left, b.left))
        qy.append((a.right, b.right))
    return _pack("Same tree — level order", "p = [1, 2, 3], q = [1, 2]", _keep(steps), ans)


def _invert_rec():
    src = [4, 2, 7, 1, 3, 6, 9]

    def build(log):
        root = _level(src)

        def walk(u, name):
            log.call(name)
            if u is None:
                return log.base(0, f"{name} = empty")
            walk(u.right, f"{u.val}R")
            walk(u.left, f"{u.val}L")
            u.left, u.right = u.right, u.left
            return log.ret(u.val, f"{name} = swapped")

        walk(root, "4")
        return _to_level(root)

    ans, steps = run_call(build)
    return _pack("Invert — recursion", "[4, 2, 7, 1, 3, 6, 9]", steps, ans)


def _invert_bfs():
    root = _level([4, 2, 7, 1, 3, 6, 9])
    q = [root]
    steps = []
    h = 0
    while h < len(q):
        u = q[h]
        h += 1
        left = "·" if u.left is None else str(u.left.val)
        right = "·" if u.right is None else str(u.right.val)
        u.left, u.right = u.right, u.left
        steps.append(line_step(
            f"swap {u.val}",
            [f"{left}, {right}  →  {right}, {left}"],
            "exchange the child pointers",
        ))
        if u.left:
            q.append(u.left)
        if u.right:
            q.append(u.right)
    ans = _to_level(root)
    steps.append(line_step("level order", [show(ans)], "mirror"))
    return _pack("Invert — level order", "[4, 2, 7, 1, 3, 6, 9]", _keep(steps), ans)


def _sum_rem():
    root = _level([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1])

    def build(log):
        def walk(u, t, name):
            log.call(name)
            if u is None:
                return log.base(False, f"{name} = False")
            if u.left is None and u.right is None:
                return log.base(u.val == t, f"{name} leaf {u.val} == {t}")
            left = walk(u.left, t - u.val, f"{u.val}L")
            if left:
                return log.ret(True, f"{name} = True")
            right = walk(u.right, t - u.val, f"{u.val}R")
            return log.ret(bool(right), f"{name} = {show(bool(right))}")

        return walk(root, 22, "5:22")

    ans, steps = run_call(build)
    return _pack("Path sum — remaining", "target 22", steps, ans)


def _sum_stack():
    root = _level([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1])
    st = [(root, 22)]
    steps = []
    ans = False
    while st:
        u, t = st.pop()
        hit = u.left is None and u.right is None and u.val == t
        steps.append(line_step(
            f"pop {u.val}, need {t}",
            ["leaf match" if hit else f"still need {t - u.val} below"],
            "True" if hit else "keep searching",
        ))
        if hit:
            ans = True
            break
        if u.left:
            st.append((u.left, t - u.val))
        if u.right:
            st.append((u.right, t - u.val))
    return _pack("Path sum — stack", "target 22", _keep(steps), ans)


def _paths_shared():
    root = _level([1, 2, 3, None, 5])
    out, path = [], []
    steps = []

    def walk(u):
        path.append(str(u.val))
        steps.append(line_step(
            f"at {u.val}",
            [" → ".join(path)],
            "emit" if u.left is None and u.right is None else "go down",
        ))
        if u.left is None and u.right is None:
            out.append("->".join(path))
        if u.left:
            walk(u.left)
        if u.right:
            walk(u.right)
        path.pop()

    walk(root)
    steps.append(line_step("emitted", list(out), "two leaves"))
    return _pack("Paths — shared list", "[1, 2, 3, null, 5]", _keep(steps), out)


def _paths_copy():
    root = _level([1, 2, 3, None, 5])
    out = []
    steps = []

    def walk(u, text):
        text = str(u.val) if not text else text + "->" + str(u.val)
        leaf = u.left is None and u.right is None
        steps.append(line_step(f"at {u.val}", [text], "emit" if leaf else "copy downward"))
        if leaf:
            out.append(text)
            return
        if u.left:
            walk(u.left, text)
        if u.right:
            walk(u.right, text)

    walk(root, "")
    return _pack("Paths — by value", "[1, 2, 3, null, 5]", _keep(steps), out)


def _sizes_climb():
    parent = [-1, 0, 0, 1, 1, 1]
    s = "abaabc"
    n = len(parent)
    new_par = parent[:]
    steps = []
    for x in range(1, n):
        v = parent[x]
        seen = [str(x)]
        while v != -1 and s[v] != s[x]:
            seen.append(f"{v}{s[v]}")
            v = parent[v]
        if v != -1:
            new_par[x] = v
            seen.append(f"stop at {v}")
        else:
            seen.append("no match")
        steps.append(line_step(
            f"{x}{s[x]}",
            [" ".join(seen), f"parent {new_par[x]}"],
            f"new parent {new_par[x]}",
        ))
    sz = [0] * n
    for x in range(n):
        v = x
        while v != -1:
            sz[v] += 1
            v = new_par[v]
    steps.append(row_step("subtree sizes", [("sz", sz, None)], show(sz)))
    return _pack("Subtree sizes — climb", 'parent [-1, 0, 0, 1, 1, 1], "abaabc"', _keep(steps), sz)


def _sizes_cell():
    parent = [-1, 0, 0, 1, 1, 1]
    s = "abaabc"
    n = len(parent)
    ch = [[] for _ in range(n)]
    for i in range(1, n):
        ch[parent[i]].append(i)
    new_par = parent[:]
    order = []
    last = [-1] * 26
    enter = -2
    st = [(0, enter)]
    steps = []
    while st:
        u, tag = st.pop()
        c = ord(s[u]) - 97
        if tag != enter:
            last[c] = tag
            continue
        old = new_par[u]
        if last[c] != -1:
            new_par[u] = last[c]
        steps.append(line_step(
            f"enter {u}{s[u]}",
            [f"last[{s[u]}] = {show(last[c])}", f"parent {old} → {new_par[u]}"],
            f"new parent {new_par[u]}",
        ))
        st.append((u, last[c]))
        last[c] = u
        order.append(u)
        for v in reversed(ch[u]):
            st.append((v, enter))
    sz = [1] * n
    for k in range(n - 1, 0, -1):
        sz[new_par[order[k]]] += sz[order[k]]
    steps.append(row_step("subtree sizes", [("sz", sz, None)], show(sz)))
    return _pack("Subtree sizes — one cell per letter", 'parent [-1, 0, 0, 1, 1, 1], "abaabc"', _keep(steps), sz)


def _spruce_run(which):
    n, ps = 7, [1, 1, 1, 2, 2, 2]
    if which == "count":
        children = [0] * (n + 1)
        leaf_kids = [0] * (n + 1)
        p = [0] * (n + 1)
        for i, pi in enumerate(ps, start=2):
            p[i] = pi
            children[pi] += 1
        for i in range(2, n + 1):
            if children[i] == 0:
                leaf_kids[p[i]] += 1
        steps = [
            row_step("children", [("ch", children[1:], None)], "pass 1"),
            row_step("leaf-children", [("leaf", leaf_kids[1:], 0)], "vertex 1 has 2"),
        ]
        ok = all(leaf_kids[v] >= 3 for v in range(1, n + 1) if children[v])
        ans = "Yes" if ok else "No"
        steps.append(chip_step("vertex 1", [("leaf-children", leaf_kids[1], "red")], ans))
    else:
        ch = [[] for _ in range(n + 1)]
        for i, pi in enumerate(ps, start=2):
            ch[pi].append(i)
        steps = [line_step(
            "children",
            [f"{v}: {ch[v]}" for v in range(1, n + 1) if ch[v]],
            "a leaf has an empty list",
        )]
        ans = "Yes"
        for v in range(1, n + 1):
            if not ch[v]:
                continue
            leaves = [c for c in ch[v] if not ch[c]]
            steps.append(line_step(
                f"vertex {v}",
                [f"children {ch[v]}", f"leaf-children {leaves}"],
                "No" if len(leaves) < 3 else "enough",
            ))
            if len(leaves) < 3:
                ans = "No"
                break
    title = "Spruce — two counters" if which == "count" else "Spruce — children lists"
    return _pack(title, "n = 7, parents 1 1 1 2 2 2", steps, ans)


def _star_deg():
    n = 5
    edges = [(1, 4), (2, 4), (3, 4), (4, 5)]
    deg = [0] * (n + 1)
    steps = []
    for i, (u, v) in enumerate(edges):
        deg[u] += 1
        deg[v] += 1
        steps.append(row_step(
            f"edge {u}–{v}",
            [("deg", deg[1:], deg[1:].index(max(deg[1:])) if i == len(edges) - 1 else None)],
            f"max = {show(max(deg[1:]))}",
        ))
    ans = "Yes" if max(deg[1:]) == n - 1 else "No"
    return _pack("Star — degrees", "5 vertices, edges 1–4, 2–4, 3–4, 4–5", steps, ans)


def _star_cand():
    n = 5
    edges = [(1, 4), (2, 4), (3, 4), (4, 5)]
    a, b = edges[0]
    ca = cb = 0
    steps = [chip_step("first edge", [("a", a, "amber"), ("b", b, "indigo")], "the centre is one of these")]
    for u, v in edges[1:]:
        if u == a or v == a:
            ca += 1
        elif u == b or v == b:
            cb += 1
        steps.append(chip_step(
            f"edge {u}–{v}",
            [("touch a", ca, "amber"), ("touch b", cb, "green" if cb == n - 2 else "indigo")],
            f"need {n - 2}",
        ))
    ans = "Yes" if ca == n - 2 or cb == n - 2 else "No"
    return _pack("Star — two candidates", "5 vertices, edges 1–4, 2–4, 3–4, 4–5", steps, ans)


def _mex_labels(mode):
    n = 6
    edges = [(1, 2), (1, 3), (2, 4), (2, 5), (5, 6)]
    deg = [0] * (n + 1)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    label = [-1] * (n - 1)
    nxt = 0
    steps = []
    if mode == "hub":
        hub = next((x for x in range(n, 0, -1) if deg[x] >= 3), None)
        steps.append(chip_step("hub", [("vertex", hub, "green"), ("degree", deg[hub], "indigo")], "degree at least 3"))
        for i, (u, v) in enumerate(edges):
            if nxt >= 3:
                break
            if u == hub or v == hub:
                label[i] = nxt
                nxt += 1
                steps.append(row_step(f"label {label[i]} on edge {i}", [("lab", [show(x) for x in label], i)], f"edge {u}–{v}"))
    else:
        leaves = sum(1 for x in range(1, n + 1) if deg[x] == 1)
        steps.append(chip_step("leaves", [("count", leaves, "green")], "at least three"))
        for i, (u, v) in enumerate(edges):
            if nxt >= 3:
                break
            if deg[u] == 1 or deg[v] == 1:
                label[i] = nxt
                nxt += 1
                steps.append(row_step(f"label {label[i]} on edge {i}", [("lab", [show(x) for x in label], i)], f"edge {u}–{v}"))
    for i in range(n - 1):
        if label[i] == -1:
            label[i] = nxt
            nxt += 1
    steps.append(row_step("labels in input order", [("lab", label, None)], "largest path MEX = 2"))
    want = [0, 3, 1, 2, 4] if mode == "hub" else [3, 0, 1, 4, 2]
    if label != want or _largest_mex(n, edges, label) != 2:
        raise AssertionError((mode, label))
    title = "MEX — at a hub" if mode == "hub" else "MEX — on leaf edges"
    return _pack(title, "edges 1–2, 1–3, 2–4, 2–5, 5–6", _keep(steps), 2)


def _largest_mex(n, edges, label):
    adj = [[] for _ in range(n + 1)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, label[i]))
        adj[v].append((u, label[i]))
    best = 0
    for s in range(1, n + 1):
        st = [(s, -1, 0)]
        while st:
            u, parent, mask = st.pop()
            for v, lab in adj[u]:
                if v == parent:
                    continue
                nm = mask | (1 << lab)
                mex = 0
                while nm & (1 << mex):
                    mex += 1
                best = max(best, mex)
                st.append((v, u, nm))
    return best


def _transit_dist(n, edges, k):
    adj = [[] for _ in range(n + 1)]
    for a, b, c in edges:
        adj[a].append((b, c))
        adj[b].append((a, c))
    dist = [-1] * (n + 1)
    dist[k] = 0
    order = [k]
    frames = []
    h = 0
    while h < len(order):
        u = order[h]
        h += 1
        grew = False
        for v, w in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + w
                order.append(v)
                grew = True
        if grew or u == k:
            frames.append(row_step(
                f"from {u}",
                [("dist", [show(x) for x in dist[1:]], None)],
                f"dist[{u}] = {show(dist[u])}",
            ))
    return dist, frames


def _transit_bfs():
    n = 5
    edges = [(1, 2, 1), (1, 3, 1), (2, 4, 1), (3, 5, 1)]
    queries = [(2, 4), (2, 3), (4, 5)]
    dist, frames = _transit_dist(n, edges, 1)
    ans = [dist[x] + dist[y] for x, y in queries]
    frames.append(line_step(
        "three queries",
        [f"({x},{y}) → {show(dist[x] + dist[y])}" for x, y in queries],
        show(ans),
    ))
    return _pack("Transit — one BFS", "K = 1, queries (2,4), (2,3), (4,5)", _keep(frames), ans)


def _transit_slow():
    n = 5
    edges = [(1, 2, 1), (1, 3, 1), (2, 4, 1), (3, 5, 1)]
    queries = [(2, 4), (2, 3), (4, 5)]
    ans = []
    steps = []
    for x, y in queries:
        dist, _frames = _transit_dist(n, edges, 1)
        ans.append(dist[x] + dist[y])
        steps.append(line_step(
            f"search again for ({x},{y})",
            [f"dist[{x}] + dist[{y}] = {show(dist[x])} + {show(dist[y])}"],
            show(dist[x] + dist[y]),
        ))
    return _pack("Transit — search per query", "K = 1, queries (2,4), (2,3), (4,5)", steps, ans)


def _infect_tree():
    return _level([1, 5, 3, None, 4, 10, 6, 9, 2])


def _infect_map():
    root = _infect_tree()
    parent = {root: None}
    source = None
    stack = [root]
    while stack:
        u = stack.pop()
        if u.val == 3:
            source = u
        for c in (u.left, u.right):
            if c is not None:
                parent[c] = u
                stack.append(c)
    dist = {source: 0}
    q = [source]
    steps = []
    best = 0
    h = 0
    while h < len(q):
        u = q[h]
        h += 1
        best = max(best, dist[u])
        born = []
        for v in (u.left, u.right, parent[u]):
            if v is not None and v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
                born.append(f"{v.val}@{dist[v]}")
        if born or u is source:
            steps.append(line_step(
                f"minute {dist[u]}",
                [f"at {u.val}"] + (born or ["no new node"]),
                f"best {best}",
            ))
    return _pack("Infection — parent map", "start at 3", _keep(steps), best)


def _infect_up():
    root = _infect_tree()

    def build(log):
        best = {"v": 0}

        def go(u, name):
            log.call(name)
            if u is None:
                return log.base(0, f"{name} = 0")
            left = go(u.left, name + "L")
            right = go(u.right, name + "R")
            if u.val == 3:
                best["v"] = max(best["v"], left, right)
                return log.ret(-1, f"start, best {best['v']}")
            if left >= 0 and right >= 0:
                return log.ret(max(left, right) + 1, f"{name} height {max(left, right) + 1}")
            down = -min(left, right)
            best["v"] = max(best["v"], down + max(left, right))
            return log.ret(min(left, right) - 1, f"best {best['v']}")

        go(root, "1")
        return best["v"]

    ans, steps = run_call(build)
    return _pack("Infection — one postorder", "start at 3", steps, ans)


def _stories():
    parts = [
        _count_walk(), _count_perfect(), _count_bisect(),
        _min_rec(), _min_bfs(),
        _same_rec(), _same_bfs(),
        _invert_rec(), _invert_bfs(),
        _sum_rem(), _sum_stack(),
        _paths_shared(), _paths_copy(),
        _sizes_climb(), _sizes_cell(),
        _spruce_run("count"), _spruce_run("lists"),
        _star_deg(), _star_cand(),
        _mex_labels("hub"), _mex_labels("leaf"),
        _transit_bfs(), _transit_slow(),
        _infect_map(), _infect_up(),
    ]
    keys = [
        ("count", "walk"), ("count", "perfect"), ("count", "bisect"),
        ("min", "rec"), ("min", "bfs"),
        ("same", "rec"), ("same", "bfs"),
        ("invert", "rec"), ("invert", "bfs"),
        ("sum", "rem"), ("sum", "stack"),
        ("paths", "shared"), ("paths", "copy"),
        ("sizes", "climb"), ("sizes", "cell"),
        ("spruce", "count"), ("spruce", "lists"),
        ("star", "deg"), ("star", "cand"),
        ("mex", "hub"), ("mex", "leaf"),
        ("transit", "bfs"), ("transit", "slow"),
        ("infect", "map"), ("infect", "up"),
    ]
    expect = {
        ("count", "walk"): "6",
        ("count", "perfect"): "6",
        ("count", "bisect"): "6",
        ("min", "rec"): "2",
        ("min", "bfs"): "2",
        ("same", "rec"): "False",
        ("same", "bfs"): "False",
        ("invert", "rec"): "[4, 7, 2, 9, 6, 3, 1]",
        ("invert", "bfs"): "[4, 7, 2, 9, 6, 3, 1]",
        ("sum", "rem"): "True",
        ("sum", "stack"): "True",
        ("paths", "shared"): "['1->2->5', '1->3']",
        ("paths", "copy"): "['1->2->5', '1->3']",
        ("sizes", "climb"): "[6, 3, 1, 1, 1, 1]",
        ("sizes", "cell"): "[6, 3, 1, 1, 1, 1]",
        ("spruce", "count"): "No",
        ("spruce", "lists"): "No",
        ("star", "deg"): "Yes",
        ("star", "cand"): "Yes",
        ("mex", "hub"): "2",
        ("mex", "leaf"): "2",
        ("transit", "bfs"): "[3, 2, 4]",
        ("transit", "slow"): "[3, 2, 4]",
        ("infect", "map"): "4",
        ("infect", "up"): "4",
    }
    out = {}
    for key, story in zip(keys, parts):
        if story[3] != expect[key]:
            raise AssertionError(f"{key}: {story[3]!r} != {expect[key]!r}")
        out[key] = story
    return out


def get(problem, form):
    """Return ``(title, input line, steps, final)`` for one solution form."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _stories()
    return _CACHE[(problem, form)]
