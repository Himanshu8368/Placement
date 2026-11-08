"""Step boards for the subsequence DP chapter (the LIS family)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import chip_step, line_step, row_step, run_call, show

LIS = [10, 9, 2, 5, 3, 7, 101, 18]
NLIS = [1, 3, 5, 4, 7]
DOLLS = [[5, 4], [6, 4], [6, 7], [2, 3]]
CUBES = [[50, 45, 20], [95, 37, 53], [45, 23, 12]]
MOUNT = [2, 1, 1, 5, 6, 2, 3, 1]
ARITH = [3, 6, 9, 12]
LIS2 = ([4, 2, 1, 4, 3, 4, 5, 8, 15], 3)
KINC = ([5, 4, 3, 2, 1], 1)
MOD3 = [3, 6, 5, 1, 8]
DIST = "abc"


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    steps = list(steps)
    if len(steps) > 8:
        steps = steps[:7] + steps[-1:]
    return (title, inp, steps, text)


def lis_tab(a):
    n = len(a)
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if a[j] < a[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return dp


def lis_tails(a):
    tails = []
    snaps = []
    for x in a:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] >= x:
                hi = mid
            else:
                lo = mid + 1
        if lo == len(tails):
            tails.append(x)
        else:
            tails[lo] = x
        snaps.append(list(tails))
    return snaps


def nlis_tab(a):
    n = len(a)
    length = [1] * n
    count = [1] * n
    for i in range(n):
        for j in range(i):
            if a[j] < a[i]:
                if length[j] + 1 > length[i]:
                    length[i] = length[j] + 1
                    count[i] = count[j]
                elif length[j] + 1 == length[i]:
                    count[i] += count[j]
    best = max(length)
    return length, count, sum(count[i] for i in range(n) if length[i] == best)


def doll_order(env):
    return sorted(env, key=lambda e: (e[0], -e[1]))


def doll_lis(env):
    order = doll_order(env)
    heights = [h for _w, h in order]
    return order, lis_tab(heights)


def cubes_norm(cuboids):
    boxes = [tuple(sorted(c)) for c in cuboids]
    boxes.sort()
    return boxes


def cubes_tab(cuboids):
    boxes = cubes_norm(cuboids)
    n = len(boxes)
    dp = [b[2] for b in boxes]
    for i in range(n):
        for j in range(i):
            if all(boxes[j][t] <= boxes[i][t] for t in range(3)):
                dp[i] = max(dp[i], dp[j] + boxes[i][2])
    return boxes, dp


def mountain(a):
    left = lis_tab(a)
    right = lis_tab(a[::-1])[::-1]
    best = 0
    for i in range(len(a)):
        if left[i] >= 2 and right[i] >= 2:
            best = max(best, left[i] + right[i] - 1)
    return left, right, (0 if best == 0 else len(a) - best)


def arith_tab(a):
    best = 1
    rows = []
    dp = [{} for _ in a]
    for i, x in enumerate(a):
        for j in range(i):
            d = x - a[j]
            dp[i][d] = dp[j].get(d, 1) + 1
            best = max(best, dp[i][d])
        rows.append(dict(dp[i]))
    return rows, best


def lis2(nums, k):
    m = max(nums)
    size = 1
    while size < m + 2:
        size *= 2
    tree = [0] * (2 * size)

    def qry(lo, hi):
        if lo > hi:
            return 0
        lo += size
        hi += size
        ans = 0
        while lo <= hi:
            if lo & 1:
                ans = max(ans, tree[lo])
                lo += 1
            if not hi & 1:
                ans = max(ans, tree[hi])
                hi -= 1
            lo //= 2
            hi //= 2
        return ans

    def upd(i, val):
        i += size
        tree[i] = max(tree[i], val)
        i //= 2
        while i:
            tree[i] = max(tree[2 * i], tree[2 * i + 1])
            i //= 2

    dp = []
    for x in nums:
        got = qry(max(1, x - k), x - 1) + 1
        upd(x, got)
        dp.append(got)
    return dp


def lnds(a):
    tails = []
    for x in a:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] > x:
                hi = mid
            else:
                lo = mid + 1
        if lo == len(tails):
            tails.append(x)
        else:
            tails[lo] = x
    return len(tails)


def k_increasing(arr, k):
    changes = 0
    chains = []
    for r in range(k):
        chain = arr[r::k]
        chains.append(chain)
        changes += len(chain) - lnds(chain)
    return chains, changes


def mod3(nums):
    NEG = -(10**18)
    best = [0, NEG, NEG]
    snaps = [list(best)]
    for x in nums:
        nxt = list(best)
        for r in range(3):
            if best[r] == NEG:
                continue
            s = (r + x) % 3
            nxt[s] = max(nxt[s], best[r] + x)
        best = nxt
        snaps.append(list(best))
    return snaps, best[0]


def distinct(s):
    end = {}
    total = 1
    snaps = []
    for ch in s:
        old = end.get(ch, 0)
        end[ch] = total
        total = total + total - old
        snaps.append((ch, dict(end), total - 1))
    return snaps, total - 1


def _lis_rows(which):
    a = LIS
    if which == "tab":
        dp = [1] * len(a)
        steps = [row_step("start", [("a", a, None), ("dp", dp, None)], "every index is a subsequence of length 1")]
        for i in range(len(a)):
            for j in range(i):
                if a[j] < a[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
            if i in (2, 4, 5, 6, 7):
                steps.append(row_step(f"i = {i}", [("a", a, i), ("dp", list(dp), i)], f"dp[{i}] = {dp[i]}"))
        assert max(dp) == 4
        return _pack("LIS — table", "[10, 9, 2, 5, 3, 7, 101, 18]", steps, max(dp))
    snaps = lis_tails(a)
    steps = []
    for i, tails in enumerate(snaps):
        if i in (0, 2, 4, 5, 6, 7):
            steps.append(row_step(f"place {a[i]}", [("tails", tails, None)], f"length {len(tails)}"))
    assert len(snaps[-1]) == 4
    return _pack("LIS — tails", "[10, 9, 2, 5, 3, 7, 101, 18]", steps, len(snaps[-1]))


def _lis_rec(kind):
    a = LIS

    def build(log):
        memo = {}

        def f(i):
            label = f"f({i})"
            if kind == "memo" and i in memo:
                return log.hit(label, memo[i])
            log.call(label)
            best = 1
            for j in range(i):
                if a[j] < a[i]:
                    best = max(best, 1 + f(j))
            if kind == "memo":
                memo[i] = best
            return log.ret(best, f"f({i}) = {best}")

        return max(f(i) for i in range(len(a)))

    answer, steps = run_call(build)
    assert answer == 4
    title = "LIS — recursion" if kind == "rec" else "LIS — memo"
    return _pack(title, "[10, 9, 2, 5, 3, 7, 101, 18]", steps, answer)


def _nlis(kind):
    a = NLIS
    if kind in ("rec", "memo"):
        def build(log):
            memo = {}

            def f(i):
                label = f"f({i})"
                if kind == "memo" and i in memo:
                    return log.hit(label, memo[i])
                log.call(label)
                length, count = 1, 1
                for j in range(i):
                    if a[j] < a[i]:
                        lj, cj = f(j)
                        if lj + 1 > length:
                            length, count = lj + 1, cj
                        elif lj + 1 == length:
                            count += cj
                if kind == "memo":
                    memo[i] = (length, count)
                return log.ret((length, count), f"f({i}) = {length},{count}")

            pairs = [f(i) for i in range(len(a))]
            best = max(p[0] for p in pairs)
            return sum(p[1] for p in pairs if p[0] == best)

        answer, steps = run_call(build)
        assert answer == 2
        title = "Number of LIS — recursion" if kind == "rec" else "Number of LIS — memo"
        return _pack(title, "[1, 3, 5, 4, 7]", steps, answer)
    length, count, answer = nlis_tab(a)
    steps = []
    L, C = [1] * len(a), [1] * len(a)
    steps.append(row_step("start", [("a", a, None), ("len", L, None), ("cnt", C, None)]))
    for i in range(len(a)):
        for j in range(i):
            if a[j] < a[i]:
                if L[j] + 1 > L[i]:
                    L[i], C[i] = L[j] + 1, C[j]
                elif L[j] + 1 == L[i]:
                    C[i] += C[j]
        steps.append(row_step(f"i = {i}", [("len", list(L), i), ("cnt", list(C), i)], f"{L[i]} ways × {C[i]}"))
    assert answer == 2 and L == length and C == count
    return _pack("Number of LIS — pairs", "[1, 3, 5, 4, 7]", steps, answer)


def _doll(kind):
    order, heights_dp = doll_lis(DOLLS)
    heights = [h for _w, h in order]
    assert max(heights_dp) == 3
    if kind in ("rec", "memo"):
        def build(log):
            memo = {}

            def f(i):
                label = f"f({i})"
                if kind == "memo" and i in memo:
                    return log.hit(label, memo[i])
                log.call(label)
                best = 1
                for j in range(i):
                    if heights[j] < heights[i]:
                        best = max(best, 1 + f(j))
                if kind == "memo":
                    memo[i] = best
                return log.ret(best)

            return max(f(i) for i in range(len(heights)))

        answer, steps = run_call(build)
        assert answer == 3
        title = "Dolls — recursion" if kind == "rec" else "Dolls — memo"
        return _pack(title, "widths up, heights down", steps, answer)
    if kind == "tab":
        steps = [row_step("sorted heights", [("h", heights, None), ("dp", heights_dp, None)], "strict LIS on heights")]
        steps.append(line_step("chain", ["[2, 3] ⊂ [5, 4] ⊂ [6, 7]"], "equal width never nests"))
        return _pack("Dolls — table", "[[5,4],[6,4],[6,7],[2,3]]", steps, 3)
    snaps = lis_tails(heights)
    steps = [row_step(f"tails {i}", [("tails", t, None)]) for i, t in enumerate(snaps)]
    return _pack("Dolls — tails", "heights after the sort", steps, len(snaps[-1]))


def _cube(kind):
    boxes, dp = cubes_tab(CUBES)
    assert max(dp) == 190
    labels = [f"{b[0]},{b[1]},{b[2]}" for b in boxes]
    if kind == "rec":
        def build(log):
            def f(i):
                log.call(f"f({i})")
                best = boxes[i][2]
                for j in range(i):
                    if all(boxes[j][t] <= boxes[i][t] for t in range(3)):
                        best = max(best, f(j) + boxes[i][2])
                return log.ret(best)

            return max(f(i) for i in range(len(boxes)))

        answer, steps = run_call(build)
        assert answer == 190
        return _pack("Cuboids — recursion", "three boxes", steps, answer)
    steps = [
        row_step("normalised", [("box", labels, None), ("h", [b[2] for b in boxes], None)]),
        row_step("height sum", [("dp", dp, len(dp) - 1)], "45 + 50 + 95"),
    ]
    title = {"memo": "Cuboids — memo", "tab": "Cuboids — table", "sort": "Cuboids — sorted chain"}[kind]
    return _pack(title, "[[50,45,20],[95,37,53],[45,23,12]]", steps, max(dp))


def _mount(kind):
    left, right, answer = mountain(MOUNT)
    assert answer == 3
    if kind in ("rec", "memo"):
        a = MOUNT

        def build(log):
            memo = {}

            def f(i):
                label = f"L({i})"
                if kind == "memo" and label in memo:
                    return log.hit(label, memo[label])
                log.call(label)
                best = 1
                for j in range(i):
                    if a[j] < a[i]:
                        best = max(best, 1 + f(j))
                if kind == "memo":
                    memo[label] = best
                return log.ret(best)

            f(4)
            return answer

        _ans, steps = run_call(build)
        title = "Mountain — left LIS" if kind == "rec" else "Mountain — memo"
        return _pack(title, "[2, 1, 1, 5, 6, 2, 3, 1]", steps, answer)
    steps = [
        row_step("from the left", [("a", MOUNT, 4), ("left", left, 4)]),
        row_step("from the right", [("right", right, 4)], "peak 6: left 3 + right 3 − 1 = 5"),
        line_step("removals", ["8 − 5 = 3"], "both arms must have length at least 2"),
    ]
    title = "Mountain — two tables" if kind == "tab" else "Mountain — tails"
    return _pack(title, "[2, 1, 1, 5, 6, 2, 3, 1]", steps, answer)


def _arith(kind):
    rows, answer = arith_tab(ARITH)
    assert answer == 4
    if kind in ("rec", "memo"):
        a = ARITH

        def build(log):
            memo = {}

            def f(i, d):
                label = f"f({i},{d})"
                if kind == "memo" and (i, d) in memo:
                    return log.hit(label, memo[(i, d)])
                log.call(label)
                best = 2
                for j in range(i):
                    if a[i] - a[j] == d:
                        best = max(best, 1 + (f(j, d) if j else 1))
                        if j == 0:
                            best = max(best, 2)
                if kind == "memo":
                    memo[(i, d)] = best
                return log.ret(best)

            return f(3, 3)

        got, steps = run_call(build)
        assert got == 4
        title = "Arithmetic — recursion" if kind == "rec" else "Arithmetic — memo"
        return _pack(title, "[3, 6, 9, 12], step 3", steps, got)
    steps = []
    for i, row in enumerate(rows):
        if not row:
            continue
        steps.append(chip_step(f"i = {i}", [(f"d={d}", row[d], "indigo") for d in sorted(row)], f"a[{i}] = {ARITH[i]}"))
    return _pack("Arithmetic — steps", "[3, 6, 9, 12]", steps, answer)


def _lis2(kind):
    nums, k = LIS2
    dp = lis2(nums, k)
    assert max(dp) == 5
    if kind in ("rec", "memo"):
        def build(log):
            memo = {}

            def f(i):
                label = f"f({i})"
                if kind == "memo" and i in memo:
                    return log.hit(label, memo[i])
                log.call(label)
                best = 1
                for j in range(i):
                    if 0 < nums[i] - nums[j] <= k:
                        best = max(best, 1 + f(j))
                if kind == "memo":
                    memo[i] = best
                return log.ret(best)

            return max(f(i) for i in range(len(nums)))

        answer, steps = run_call(build)
        assert answer == 5
        title = "LIS II — recursion" if kind == "rec" else "LIS II — memo"
        return _pack(title, "k = 3", steps, answer)
    steps = [row_step("value window", [("a", nums, None), ("dp", dp, None)], "best in [x−k, x−1], then write x")]
    steps.append(line_step("chain", ["1, 3, 4, 5, 8"], "the step to 15 is 7, larger than k"))
    title = "LIS II — table" if kind == "tab" else "LIS II — segment tree"
    return _pack(title, "[4, 2, 1, 4, 3, 4, 5, 8, 15], k = 3", steps, max(dp))


def _kinc(kind):
    arr, k = KINC
    chains, answer = k_increasing(arr, k)
    assert answer == 4
    if kind == "rec":
        steps = [line_step("one chain", [str(arr), "non-decreasing piece has length 1", "changes = 5 − 1 = 4"])]
        return _pack("K-increasing — one chain", "k = 1", steps, answer)
    shown = chains[0]
    steps = [
        row_step(f"residue 0", [("chain", shown, None)], f"LNDS {lnds(shown)}, changes {len(shown) - lnds(shown)}"),
        line_step("sum", [f"{answer} changes"], "k chains, no edge between them"),
    ]
    titles = {"memo": "K-increasing — memo", "tab": "K-increasing — chains", "bit": "K-increasing — tails"}
    return _pack(titles[kind], "[5, 4, 3, 2, 1], k = 1", steps, answer)


def _mod(kind):
    snaps, answer = mod3(MOD3)
    assert answer == 18
    steps = []
    for i, row in enumerate(snaps):
        shown = ["·" if v < 0 else v for v in row]
        label = "empty" if i == 0 else f"+ {MOD3[i - 1]}"
        steps.append(row_step(label, [("r0 r1 r2", shown, None)]))
    title = {"rec": "Mod 3 — take or skip", "memo": "Mod 3 — memo", "tab": "Mod 3 — three sums", "roll": "Mod 3 — copy the row"}[kind]
    return _pack(title, "[3, 6, 5, 1, 8]", steps, answer)


def _dist(kind):
    snaps, answer = distinct(DIST)
    assert answer == 7
    steps = []
    for ch, end, total in snaps:
        steps.append(chip_step(
            f"letter {ch}",
            [("end " + c, end.get(c, 0), "indigo") for c in "abc"] + [("answer", total, "green")],
            "double, then remove the previous copy of this letter",
        ))
    title = {"rec": "Distinct — recursion", "memo": "Distinct — memo", "tab": "Distinct — last letter", "last": "Distinct — subtract the old end"}[kind]
    return _pack(title, '"abc"', steps, answer)


def stories():
    return {
        ("lis", "rec"): _lis_rec("rec"),
        ("lis", "memo"): _lis_rec("memo"),
        ("lis", "tab"): _lis_rows("tab"),
        ("lis", "tails"): _lis_rows("tails"),
        ("nlis", "rec"): _nlis("rec"),
        ("nlis", "memo"): _nlis("memo"),
        ("nlis", "tab"): _nlis("tab"),
        ("nlis", "bit"): _nlis("bit"),
        ("doll", "rec"): _doll("rec"),
        ("doll", "memo"): _doll("memo"),
        ("doll", "tab"): _doll("tab"),
        ("doll", "tails"): _doll("tails"),
        ("cube", "rec"): _cube("rec"),
        ("cube", "memo"): _cube("memo"),
        ("cube", "tab"): _cube("tab"),
        ("cube", "sort"): _cube("sort"),
        ("mount", "rec"): _mount("rec"),
        ("mount", "memo"): _mount("memo"),
        ("mount", "tab"): _mount("tab"),
        ("mount", "tails"): _mount("tails"),
        ("arith", "rec"): _arith("rec"),
        ("arith", "memo"): _arith("memo"),
        ("arith", "tab"): _arith("tab"),
        ("arith", "fix"): _arith("fix"),
        ("lis2", "rec"): _lis2("rec"),
        ("lis2", "memo"): _lis2("memo"),
        ("lis2", "tab"): _lis2("tab"),
        ("lis2", "tree"): _lis2("tree"),
        ("kinc", "rec"): _kinc("rec"),
        ("kinc", "memo"): _kinc("memo"),
        ("kinc", "tab"): _kinc("tab"),
        ("kinc", "bit"): _kinc("bit"),
        ("mod3", "rec"): _mod("rec"),
        ("mod3", "memo"): _mod("memo"),
        ("mod3", "tab"): _mod("tab"),
        ("mod3", "roll"): _mod("roll"),
        ("dist", "rec"): _dist("rec"),
        ("dist", "memo"): _dist("memo"),
        ("dist", "tab"): _dist("tab"),
        ("dist", "last"): _dist("last"),
    }


def get(problem, form):
    return stories()[(problem, form)]


if __name__ == "__main__":
    data = stories()
    expect = {
        "lis": "4", "nlis": "2", "doll": "3", "cube": "190", "mount": "3",
        "arith": "4", "lis2": "5", "kinc": "4", "mod3": "18", "dist": "7",
    }
    for key, (title, inp, steps, final) in data.items():
        assert final == expect[key[0]], (key, final)
        assert 1 <= len(steps) <= 8, (key, len(steps))
        print(f"{key[0]:6} {key[1]:6} steps={len(steps)} final={final}")
    assert mountain([1, 3, 1])[2] == 0
    assert k_increasing([4, 1, 5, 2, 6, 2], 2)[1] == 0
    assert k_increasing([4, 1, 5, 2, 6, 2], 3)[1] == 2
    assert distinct("aba")[1] == 6 and distinct("aaa")[1] == 3
    assert mod3([4])[1] == 0 and mod3([1, 2, 3, 4, 4])[1] == 12
    assert max(cubes_tab([[38, 25, 45], [76, 35, 3]])[1]) == 76
    assert max(lis2([7, 4, 5, 1, 8, 12, 4, 7], 5)) == 4
    assert max(lis2([1, 5], 1)) == 1
    assert max(doll_lis([[1, 1], [1, 1], [1, 1]])[1]) == 1
    print("keys", len(data))
