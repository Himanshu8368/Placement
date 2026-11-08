"""Step frames for the traversal chapter.

Drawing lives in ``dsa_viz.story``. ``get`` returns data only.
Each story is the algorithm on the sample that chapter already checks,
cut down to at most eight frames.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import chip_step, line_step, row_step, run_call, show


class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def _from_level(vals):
    if not vals or vals[0] is None:
        return None
    root = Node(vals[0])
    q = [root]
    i = 1
    while q and i < len(vals):
        u = q.pop(0)
        if i < len(vals) and vals[i] is not None:
            u.left = Node(vals[i])
            q.append(u.left)
        i += 1
        if i < len(vals) and vals[i] is not None:
            u.right = Node(vals[i])
            q.append(u.right)
        i += 1
    return root


def _to_level(root):
    if root is None:
        return []
    q = [root]
    out = []
    while q:
        u = q.pop(0)
        if u is None:
            out.append(None)
            continue
        out.append(u.val)
        q.append(u.left)
        q.append(u.right)
    while out and out[-1] is None:
        out.pop()
    return out


def _tree_b():
    return _from_level([0, 1, 2, 4, 5, None, 6, 7, 8, None, None, 9])


def _cap(steps, limit=8):
    if len(steps) <= limit:
        return steps
    if limit <= 2:
        return [steps[0], steps[-1]][:limit]
    keep = limit - 2
    rest = list(range(1, len(steps) - 1))
    step = len(rest) / keep
    mid = [rest[min(len(rest) - 1, int((i + 0.5) * step))] for i in range(keep)]
    idx = sorted({0, len(steps) - 1, *mid})
    return [steps[i] for i in idx]


def _pack(title, inp, steps, answer):
    steps = _cap(list(steps))
    if not steps or len(steps) > 8:
        raise RuntimeError(f"{title}: {len(steps)} frames")
    text = answer if isinstance(answer, str) else show(answer) if not isinstance(answer, (list, tuple)) else answer
    # Lists keep Python's own str, which the chapter asserts.
    if isinstance(answer, (list, tuple)):
        text = answer
    return (title, inp, steps, text)


def _vals(nodes):
    return " ".join(show(v) for v in nodes) if nodes else "—"


def _post_rec():
    def build(log):
        root = _tree_b()
        out = []

        def go(u):
            if u is None:
                return
            log.call(str(u.val))
            go(u.left)
            go(u.right)
            out.append(u.val)
            log.ret(u.val, f"{u.val} = emit")

        go(root)
        return out

    ans, steps = run_call(build)
    return _pack("Postorder — recursion", "tree B", steps, ans)


def _post_stack():
    root = _tree_b()
    st, cur, last, out = [], root, None, []
    steps = []
    while cur is not None or st:
        while cur is not None:
            st.append(cur)
            cur = cur.left
        top = st[-1]
        if top.right is not None and top.right is not last:
            cur = top.right
        else:
            out.append(top.val)
            last = top
            st.pop()
            steps.append(line_step(
                f"emit {top.val}",
                [f"stack  {_vals(v.val for v in st)}", f"last  {top.val}", f"out  {_vals(out)}"],
                "right child missing, or it is last",
            ))
    return _pack("Postorder — one stack", "tree B", steps, out)


def _dfs_sim():
    n, p = 7, [1, 2, 5, 4, 3, 6, 7]
    st = [1]
    done = [0] * (n + 1)

    def kids(v):
        return (2 * v <= n) + (2 * v + 1 <= n)

    steps = [line_step("start at 1", [f"path  {_vals(st)}"], "p1 = 1")]
    ok = True
    for x in p[1:]:
        while st and st[-1] != x // 2:
            if done[st[-1]] != kids(st[-1]):
                ok = False
                steps.append(line_step(
                    f"reject {x}",
                    [f"path  {_vals(st)}", f"{x} wants parent {x // 2}"],
                    "parent is not on the root path",
                ))
                break
            st.pop()
        if not ok:
            break
        if not st:
            ok = False
            break
        done[x // 2] += 1
        st.append(x)
        steps.append(line_step(
            f"enter {x}",
            [f"path  {_vals(st)}", f"done[{x // 2}] = {done[x // 2]}"],
            f"parent {x // 2} is on the path",
        ))
    return _pack("DFS order — resimulate", "1 2 5 4 3 6 7", steps, ok)


def _ancestor_or_self(a, b):
    if a <= 0 or b <= 0:
        return False
    da, db = a.bit_length() - 1, b.bit_length() - 1
    return da <= db and (b >> (db - da)) == a


def _dfs_pairs():
    n, perm, swaps = 3, [1, 2, 3], [(2, 3), (3, 2), (1, 3)]
    p = list(perm)

    def good(i):
        return _ancestor_or_self(p[i + 1] // 2, p[i])

    cnt = sum(good(i) for i in range(n - 1))
    steps = [chip_step(
        "before any swap",
        [("p0", p[0], "indigo"), ("good", cnt, "amber")],
        f"need {n - 1} good pairs and p0 = 1",
    )]
    out = []
    for x, y in swaps:
        x -= 1
        y -= 1
        touched = {i for i in (x - 1, x, y - 1, y) if 0 <= i < n - 1}
        for i in touched:
            cnt -= good(i)
        p[x], p[y] = p[y], p[x]
        for i in touched:
            cnt += good(i)
        ans = "YES" if p[0] == 1 and cnt == n - 1 else "NO"
        out.append(ans)
        steps.append(chip_step(
            f"swap positions {x + 1},{y + 1}",
            [("p0", p[0], "green" if p[0] == 1 else "amber"), ("good", cnt, "indigo"), ("ans", ans, "green" if ans == "YES" else "amber")],
            f"p = {_vals(p)}",
        ))
    return _pack("DFS order — good pairs", "n = 3, swaps (2,3) (3,2) (1,3)", steps, out)


def _dfs_places():
    n, perm, swaps = 3, [1, 2, 3], [(2, 3), (3, 2), (1, 3)]
    p = list(perm)
    pos = [0] * (n + 1)
    for i, v in enumerate(p):
        pos[v] = i
    sz = [0] * (n + 2)
    for v in range(n, 0, -1):
        sz[v] = 1 + (sz[2 * v] + sz[2 * v + 1] if 2 * v <= n else 0)
    internal = n // 2

    def ok(v):
        if not 1 <= v <= internal:
            return 0
        a, b, s = pos[2 * v], pos[2 * v + 1], sz[2 * v]
        here = pos[v] + 1
        return int((a == here and b == here + s) or (b == here and a == here + s))

    cnt = sum(ok(v) for v in range(1, internal + 1))
    steps = [chip_step(
        "child slots of 1",
        [("good", cnt, "amber"), ("internal", internal, "indigo")],
        "children occupy pos[v]+1 and pos[v]+1+sz, either order",
    )]
    out = []
    for x, y in swaps:
        x, y = x - 1, y - 1
        touched = {p[x], p[x] // 2, p[y], p[y] // 2}
        for v in touched:
            cnt -= ok(v)
        p[x], p[y] = p[y], p[x]
        pos[p[x]], pos[p[y]] = x, y
        for v in touched:
            cnt += ok(v)
        ans = "YES" if p[0] == 1 and cnt == internal else "NO"
        out.append(ans)
        steps.append(chip_step(
            f"swap positions {x + 1},{y + 1}",
            [("p0", p[0], "green" if p[0] == 1 else "amber"), ("good", cnt, "indigo"), ("ans", ans, "green" if ans == "YES" else "amber")],
            f"p = {_vals(p)}",
        ))
    return _pack("DFS order — child positions", "n = 3, swaps (2,3) (3,2) (1,3)", steps, out)


def _level_rows(root):
    q = [root]
    rows = []
    while q:
        nxt, row = [], []
        for u in q:
            row.append(u.val)
            if u.left:
                nxt.append(u.left)
            if u.right:
                nxt.append(u.right)
        rows.append((q, row))
        q = nxt
    return rows


def _avg_bfs():
    root = _from_level([3, 9, 20, None, None, 15, 7])
    out, steps = [], []
    for nodes, row in _level_rows(root):
        total = sum(row)
        avg = total / len(row)
        out.append(avg)
        steps.append(row_step(
            f"sum {total} / {len(row)}",
            [("level", row, len(row) - 1)],
            f"{show(avg)} = {total} / {len(row)}",
        ))
    return _pack("Level averages — BFS", "[3,9,20,null,null,15,7]", steps, out)


def _avg_dfs():
    root = _from_level([3, 9, 20, None, None, 15, 7])
    sums, cnt = [], []
    steps = []

    def go(u, d):
        if u is None:
            return
        if len(sums) == d:
            sums.append(0)
            cnt.append(0)
        sums[d] += u.val
        cnt[d] += 1
        steps.append(line_step(
            f"visit {u.val} at depth {d}",
            [f"sum  {_vals(sums)}", f"cnt  {_vals(cnt)}"],
            f"sum[{d}] += {u.val}",
        ))
        go(u.left, d + 1)
        go(u.right, d + 1)

    go(root, 0)
    out = [sums[d] / cnt[d] for d in range(len(sums))]
    return _pack("Level averages — DFS", "[3,9,20,null,null,15,7]", steps, out)


def _zig_rev():
    root = _from_level([3, 9, 20, None, None, 15, 7])
    out, steps = [], []
    for _nodes, row in _level_rows(root):
        stored = row[::-1] if len(out) % 2 else list(row)
        steps.append(row_step(
            "reverse" if len(out) % 2 else "keep",
            [("row", stored, None)],
            "odd levels are stored from the right" if len(out) % 2 else "level 0 stays left to right",
        ))
        out.append(stored)
    return _pack("Zigzag — reverse", "[3,9,20,null,null,15,7]", steps, out)


def _zig_place():
    root = _from_level([3, 9, 20, None, None, 15, 7])
    out, steps, d = [], [], 0
    for nodes, _row in _level_rows(root):
        s = len(nodes)
        level = [None] * s
        for i, u in enumerate(nodes):
            j = i if d % 2 == 0 else s - 1 - i
            level[j] = u.val
            shown = ["·" if v is None else v for v in level]
            steps.append(row_step(
                f"write {u.val} at {j}",
                [("level", shown, j)],
                "left to right" if d % 2 == 0 else f"slot {s - 1} - {i}",
            ))
        out.append([v for v in level])
        d += 1
    return _pack("Zigzag — place", "[3,9,20,null,null,15,7]", steps, out)


def _zig_stacks():
    root = _from_level([3, 9, 20, None, None, 15, 7])
    cur, nxt = [root], []
    out, left_to_right, steps = [], True, []
    while cur:
        row = []
        while cur:
            u = cur.pop()
            row.append(u.val)
            kids = (u.left, u.right) if left_to_right else (u.right, u.left)
            for c in kids:
                if c is not None:
                    nxt.append(c)
        out.append(row)
        steps.append(line_step(
            "left to right" if left_to_right else "right to left",
            [f"row  {_vals(row)}", f"next  {_vals(v.val for v in nxt)}"],
            "push left, then right" if left_to_right else "push right, then left",
        ))
        cur, nxt = nxt, cur
        left_to_right = not left_to_right
    return _pack("Zigzag — two stacks", "[3,9,20,null,null,15,7]", steps, out)


def _cous_passes():
    root = _from_level([5, 4, 9, 1, 10, None, 7])
    root.val = 0
    cur = [root]
    steps = [line_step("root", ["root = 0"], "the root has no cousins")]
    while cur:
        next_sum = 0
        for u in cur:
            if u.left:
                next_sum += u.left.val
            if u.right:
                next_sum += u.right.val
        nxt = []
        groups = []
        for u in cur:
            group = (u.left.val if u.left else 0) + (u.right.val if u.right else 0)
            groups.append(group)
            for c in (u.left, u.right):
                if c:
                    c.val = next_sum - group
                    nxt.append(c)
        if nxt:
            steps.append(line_step(
                f"level sum {next_sum}",
                [f"groups  {_vals(groups)}", f"wrote  {_vals(v.val for v in nxt)}"],
                "child = level sum − sibling group",
            ))
        cur = nxt
    return _pack("Cousin sums — two passes", "[5,4,9,1,10,null,7]", steps, _to_level(root))


def _cous_walks():
    root = _from_level([5, 4, 9, 1, 10, None, 7])
    level_sum = []
    st = [(root, 0)]
    while st:
        u, d = st.pop()
        if len(level_sum) <= d:
            level_sum.extend([0] * (d + 1 - len(level_sum)))
        level_sum[d] += u.val
        if u.left:
            st.append((u.left, d + 1))
        if u.right:
            st.append((u.right, d + 1))
    steps = [row_step(
        "old level sums",
        [("sum", level_sum, len(level_sum) - 1)],
        "read every old value before a write",
    )]
    root.val = 0
    st = [(root, 0)]
    while st:
        u, d = st.pop()
        group = (u.left.val if u.left else 0) + (u.right.val if u.right else 0)
        wrote = []
        for c in (u.left, u.right):
            if c:
                c.val = level_sum[d + 1] - group
                wrote.append(c.val)
                st.append((c, d + 1))
        if wrote:
            steps.append(line_step(
                f"depth {d}, group {group}",
                [f"wrote  {_vals(wrote)}", f"level  {_vals(_to_level(root))}"],
                f"level sum {level_sum[d + 1]} − {group}",
            ))
    return _pack("Cousin sums — two walks", "[5,4,9,1,10,null,7]", steps, _to_level(root))


def _spine(root):
    out = []
    while root:
        out.append(root.val)
        root = root.right
    return out


_FLAT = [1, 2, 5, 3, 4, None, 6]


def _flat_collect():
    root = _from_level(_FLAT)
    st = [root]
    order = []
    steps = []
    while st:
        u = st.pop()
        order.append(u)
        if u.right:
            st.append(u.right)
        if u.left:
            st.append(u.left)
        steps.append(line_step(
            f"pop {u.val}",
            [f"order  {_vals(v.val for v in order)}", f"stack  {_vals(v.val for v in st)}"],
            "push right, then left",
        ))
    for i, node in enumerate(order):
        node.left = None
        node.right = order[i + 1] if i + 1 < len(order) else None
    steps.append(line_step("relink", [f"spine  {_vals(_spine(root))}"], "left is null, right is the next node"))
    return _pack("Flatten — collect", "[1,2,5,3,4,null,6]", steps, _spine(root))


def _flat_splice():
    root = _from_level(_FLAT)
    cur = root
    steps = []
    while cur:
        if cur.left:
            p = cur.left
            while p.right:
                p = p.right
            old = cur.right
            left_val = cur.left.val
            p.right = old
            cur.right = cur.left
            cur.left = None
            steps.append(line_step(
                f"splice at {cur.val}",
                [f"{p.val} → {old.val}", f"{cur.val}.right = {left_val}", f"{cur.val}.left = null"],
                "point the rightmost node at the old right first",
            ))
        cur = cur.right
    steps.append(line_step("spine", [_vals(_spine(root))], "every left is null"))
    return _pack("Flatten — splice", "[1,2,5,3,4,null,6]", steps, _spine(root))


def _flat_end():
    root = _from_level(_FLAT)
    prev = None
    st, cur, last = [], root, None
    steps = []
    while cur is not None or st:
        while cur is not None:
            st.append(cur)
            cur = cur.right
        top = st[-1]
        if top.left is not None and top.left is not last:
            cur = top.left
        else:
            st.pop()
            last = top
            top.right = prev
            top.left = None
            prev = top
            steps.append(line_step(
                f"finish {top.val}",
                [f"chain  {_vals(_spine(prev))}"],
                "right, then left, then the node",
            ))
    return _pack("Flatten — from the end", "[1,2,5,3,4,null,6]", steps, _spine(root))


def _inorder_list(root):
    out = []
    st, cur = [], root
    while cur is not None or st:
        while cur is not None:
            st.append(cur)
            cur = cur.left
        cur = st.pop()
        out.append(cur.val)
        cur = cur.right
    return out


def _recov_stack_clean():
    root = _from_level([1, 3, None, None, 2])
    first = second = prev = None
    st, cur = [], root
    seen = []
    steps = []
    while cur is not None or st:
        while cur is not None:
            st.append(cur)
            cur = cur.left
        cur = st.pop()
        seen.append(cur.val)
        if prev is not None and prev.val > cur.val:
            if first is None:
                first = prev
            second = cur
            steps.append(chip_step(
                f"descent {prev.val} > {cur.val}",
                [("first", first.val, "amber"), ("second", second.val, "green"), ("seen", _vals(seen), "indigo")],
                "the larger of the first descent, the smaller of the last",
            ))
        prev = cur
        cur = cur.right
    a, b = first.val, second.val
    first.val, second.val = b, a
    steps.append(line_step(
        "swap",
        [f"inorder  {_vals(_inorder_list(root))}"],
        f"{a} and {b} trade places",
    ))
    return _pack("Recover BST — stack", "[1,3,null,null,2]", steps, _inorder_list(root))


def _recov_morris():
    root = _from_level([1, 3, None, None, 2])
    first = second = prev = None
    steps = []

    def visit(u):
        nonlocal first, second, prev
        if prev is not None and prev.val > u.val:
            if first is None:
                first = prev
            second = u
            steps.append(chip_step(
                f"descent {prev.val} > {u.val}",
                [("first", first.val, "amber"), ("second", second.val, "green")],
                "same test as the stack, at a Morris visit",
            ))
        prev = u

    cur = root
    while cur:
        if cur.left is None:
            visit(cur)
            cur = cur.right
            continue
        p = cur.left
        while p.right is not None and p.right is not cur:
            p = p.right
        if p.right is None:
            p.right = cur
            steps.append(line_step(
                f"thread {p.val} → {cur.val}",
                [f"at {cur.val}"],
                "borrow the predecessor's null right child",
            ))
            cur = cur.left
        else:
            p.right = None
            steps.append(line_step(
                f"clear {p.val} → {cur.val}",
                ["thread removed"],
                "the thread is gone before the value is recorded",
            ))
            visit(cur)
            cur = cur.right
    a, b = first.val, second.val
    first.val, second.val = b, a
    steps.append(line_step(
        "swap",
        [f"inorder  {_vals(_inorder_list(root))}"],
        f"{a} and {b} trade places",
    ))
    return _pack("Recover BST — Morris", "[1,3,null,null,2]", steps, _inorder_list(root))


def _ser_pre():
    root = _from_level([1, 2, 3, None, None, 4, 5])
    out = []
    steps = []

    def enc(u):
        if u is None:
            out.append("#")
            steps.append(line_step("null", [",".join(out)], "a missing child is a token"))
            return
        out.append(str(u.val))
        steps.append(line_step(f"value {u.val}", [",".join(out)], "preorder, then the two children"))
        enc(u.left)
        enc(u.right)

    enc(root)
    text = ",".join(out)
    return _pack("Serialize — preorder", "[1,2,3,null,null,4,5]", steps, text)


def _ser_stack():
    root = _from_level([1, 2, 3, None, None, 4, 5])
    out, st = [], [root]
    steps = []
    while st:
        u = st.pop()
        if u is None:
            out.append("#")
            label = "#"
        else:
            out.append(str(u.val))
            label = str(u.val)
            st.append(u.right)
            st.append(u.left)
        pending = []
        for x in st:
            pending.append("#" if x is None else str(x.val))
        steps.append(line_step(
            f"pop {label}",
            [f"out  {','.join(out)}", f"stack  {_vals(pending)}"],
            "push right, then left, nulls included",
        ))
    return _pack("Serialize — stack", "[1,2,3,null,null,4,5]", steps, ",".join(out))


_DUP = [1, 2, 3, 4, None, 2, 4, None, None, 4]


def _dup_ids():
    root = _from_level(_DUP)
    id_of = {}
    count = [0]
    found = []
    steps = []

    def go(u):
        if u is None:
            return 0
        key = (go(u.left), u.val, go(u.right))
        if key not in id_of:
            id_of[key] = len(count)
            count.append(0)
        i = id_of[key]
        count[i] += 1
        if count[i] == 2:
            found.append(u)
        steps.append(line_step(
            f"id {i} count {count[i]}",
            [f"({key[0]}, {key[1]}, {key[2]})", "report" if count[i] == 2 else "keep"],
            "report when the count becomes 2",
        ))
        return i

    go(root)
    keys = sorted(_ser_key(u) for u in found)
    return _pack("Duplicate subtrees — ids", "[1,2,3,4,null,2,4,null,null,4]", steps, keys)


def _ser_key(u):
    out = []

    def enc(x):
        if x is None:
            out.append("#")
            return
        out.append(str(x.val))
        enc(x.left)
        enc(x.right)

    enc(u)
    return ",".join(out)


def _dup_string():
    root = _from_level(_DUP)
    seen = {}
    found = []
    steps = []

    def go(u):
        if u is None:
            return "#"
        s = f"{u.val},{go(u.left)},{go(u.right)}"
        seen[s] = seen.get(s, 0) + 1
        if seen[s] == 2:
            found.append(u)
        if u.val in (2, 4) or seen[s] == 2:
            steps.append(line_step(
                f"count {seen[s]}",
                [s, "report" if seen[s] == 2 else "keep"],
                "the key is the marked preorder of this subtree",
            ))
        return s

    go(root)
    keys = sorted(_ser_key(u) for u in found)
    return _pack("Duplicate subtrees — string key", "[1,2,3,4,null,2,4,null,null,4]", steps, keys)


def _build():
    parts = [
        (("post", "rec"), _post_rec),
        (("post", "stack"), _post_stack),
        (("dfs", "sim"), _dfs_sim),
        (("dfs", "pairs"), _dfs_pairs),
        (("dfs", "places"), _dfs_places),
        (("avg", "bfs"), _avg_bfs),
        (("avg", "dfs"), _avg_dfs),
        (("zig", "rev"), _zig_rev),
        (("zig", "place"), _zig_place),
        (("zig", "stacks"), _zig_stacks),
        (("cous", "passes"), _cous_passes),
        (("cous", "walks"), _cous_walks),
        (("flat", "collect"), _flat_collect),
        (("flat", "splice"), _flat_splice),
        (("flat", "end"), _flat_end),
        (("recov", "stack"), _recov_stack_clean),
        (("recov", "morris"), _recov_morris),
        (("ser", "pre"), _ser_pre),
        (("ser", "stack"), _ser_stack),
        (("dup", "ids"), _dup_ids),
        (("dup", "string"), _dup_string),
    ]
    cache = {}
    for key, fn in parts:
        cache[key] = fn()
    return cache


_CACHE = None


def get(problem, form):
    """Return ``(title, input line, steps, final)`` for one solution form."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _build()
    return _CACHE[(problem, form)]
