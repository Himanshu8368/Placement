"""Step frames for the sorting-as-a-tool chapter.

Drawing lives in ``dsa_viz.story``. ``get`` returns data only.
Each story walks the chapter's own sample, at most eight frames, and the
numbers come from running the algorithm - never from a hand-typed table.
A sorting frame shows the array (often the sorted form) with the current
neighbour/bucket/rank highlighted plus the rolling chips (best, gap, cost),
so the reader watches a *comparison* decide, exactly as the comparator does.

Every problem carries the full ladder: a ``brute`` frame set (the direct
definition), an ``intermediate`` frame set (naive sort, or sort plus a
linear/bisect/prefix pass that improves on the brute but has not yet earned
the pattern's *insight*), and an ``optimal`` frame set (the rung the pattern
names). ``get`` returns whichever frame set the caller asks for.
"""
from __future__ import annotations

import bisect
import math
import sys
from collections import Counter
from functools import cmp_to_key
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import chip_step, row_step, show


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    if not 1 <= len(steps) <= 8:
        raise AssertionError(f"{title}: {len(steps)} frames")
    return (title, inp, steps, text)


def _step(title, rows, chips=None, eq=""):
    """One frame: one or two array rows plus the rolling chips."""
    frame = row_step(title, rows, eq=eq)
    if chips:
        frame["chips"] = chip_step(title, chips, eq="")["chips"]
    return frame


def _cmp_concat(a, b):
    """Largest-Number comparator: put ``a`` first iff a+b beats b+a (as strings)."""
    if a + b > b + a:
        return -1
    if a + b < b + a:
        return 1
    return 0


def _mindiff_sort():
    arr = [4, 2, 1, 3]
    s = sorted(arr)
    steps = [_step("sort the array", [("sorted", s, None)], [("best diff", "-", "green")],
             eq="the closest pair is always adjacent once the array is sorted")]
    best = math.inf
    pairs = []
    for i in range(1, len(s)):
        d = s[i] - s[i - 1]
        if d < best:
            best, pairs = d, [[s[i - 1], s[i]]]
            note = " -> new minimum, restart the list"
        elif d == best:
            pairs.append([s[i - 1], s[i]])
            note = " -> ties the minimum, append the pair"
        else:
            note = " -> larger, ignore"
        steps.append(_step(f"neighbours {s[i - 1]},{s[i]}", [("sorted", s, i)],
                           [("diff", d, "amber"), ("best", best, "green")],
                           eq=f"{s[i]} - {s[i - 1]} = {d}" + note))
    assert pairs == [[1, 2], [2, 3], [3, 4]]
    return _pack("Minimum Absolute Difference - sort, then scan neighbours",
                 "arr = [4, 2, 1, 3] -> sorted [1, 2, 3, 4]", steps, str(pairs))


def _honest_sort():
    order = [3, 1, 2, 6, 4]
    s = sorted(order)
    steps = [_step("sort strengths", [("s", s, None)], [("best seam", "-", "green")],
             eq="the teams' seam |max(A) - min(B)| is one adjacent gap; hunt the smallest")]
    best = math.inf
    for i in range(1, len(s)):
        d = s[i] - s[i - 1]
        best = min(best, d)
        steps.append(_step(f"gap {s[i - 1]} | {s[i]}", [("s", s, i)],
                           [("gap", d, "amber"), ("best", best, "green")],
                           eq=f"s[{i}] - s[{i - 1}] = {d}; best seam {best}"))
    assert best == 1
    return _pack("Honest Coach - the best split is the smallest adjacent gap",
                 "s = [3, 1, 2, 6, 4] -> sorted [1, 2, 3, 4, 6]", steps, best)


def _largest_cmp():
    nums = [3, 30, 34, 5, 9]
    strs = [str(x) for x in nums]
    steps = [_step("write as strings", [("nums", strs, None)], [("order", "-", "amber")],
             eq="positional order fails (3 vs 30); compare the two concatenations")]
    for a, b in [("3", "30"), ("34", "3"), ("5", "34"), ("9", "5")]:
        ab, ba = a + b, b + a
        first = a if ab > ba else b
        steps.append(_step(f"compare {a} vs {b}", [("nums", strs, None)],
                           [(f"{a}+{b}", ab, "indigo"), (f"{b}+{a}", ba, "violet"),
                            ("first", first, "green")],
                           eq=f"'{ab}' {'>' if ab > ba else '<'} '{ba}' so {first} leads"))
    order = sorted(strs, key=cmp_to_key(_cmp_concat))
    result = "".join(order)
    steps.append(_step("sorted, then joined", [("order", order, None)],
                       [("number", result, "green")],
                       eq="leading-zero guard: an all-zero input collapses to \"0\""))
    assert result == "9534330"
    return _pack("Largest Number - a comparator on concatenated strings",
                 "nums = [3, 30, 34, 5, 9]", steps, result)


def _queue_greedy():
    people = [[7, 0], [4, 4], [7, 1], [5, 0], [6, 1], [5, 2]]

    def lab(p):
        return f"{p[0]},{p[1]}"

    order = sorted(people, key=lambda p: (-p[0], p[1]))
    out = []
    steps = [_step("sort by (-height, k)", [("queue order", [lab(p) for p in order], None)],
                   [("front", "-", "green")],
                   eq="tallest first; among equal heights, smaller k first, so short insertions never disturb tall ones")]
    for p in order:
        out.insert(p[1], p)
        steps.append(_step(f"insert {lab(p)} at {p[1]}", [("queue", [lab(q) for q in out], p[1])],
                           [("k", p[1], "amber"), ("size", len(out), "green")],
                           eq=f"exactly {p[1]} already-taller-or-equal people now sit in front"))
    assert out == [[5, 0], [7, 0], [5, 2], [6, 1], [4, 4], [7, 1]]
    return _pack("Queue Reconstruction - sort tall-first, then insert by k",
                 "people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]", steps, str(out))


def _relative_count():
    arr1 = [2, 3, 1, 3, 2, 4, 6, 7, 9, 2, 19]
    arr2 = [2, 1, 4, 3, 9, 6]
    cnt = Counter(arr1)
    in2 = set(arr2)
    rest = sorted(v for v in cnt if v not in in2)
    res = []
    steps = [_step("histogram of arr1", [("arr1", arr1, None)],
                   [("distinct", len(cnt), "amber")],
                   eq="a counting sort keyed on value: emit counts in the order arr2 dictates")]
    for v in arr2:
        res += [v] * cnt[v]
        steps.append(_step(f"emit {v} x{cnt[v]}", [("result", res, len(res) - 1)],
                           [("value", v, "indigo"), ("copies", cnt[v], "amber")],
                           eq=f"all {cnt[v]} copies of {v}, in arr2's order"))
    for v in rest:
        res += [v] * cnt[v]
    steps.append(_step("append the rest ascending", [("result", res, None)],
                       [("leftovers", rest, "violet")],
                       eq=f"values absent from arr2, sorted: {rest}"))
    assert res == [2, 2, 2, 1, 4, 3, 3, 9, 6, 7, 19]
    return _pack("Relative Sort Array - counting sort by arr2 order",
                 "arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]", steps, str(res))


def _gap_bucket():
    nums = [3, 6, 9, 1]
    n = len(nums)
    mn, mx = min(nums), max(nums)
    size = max(1, (mx - mn) // (n - 1))
    nb = (mx - mn) // size + 1
    bmin = [math.inf] * nb
    bmax = [-math.inf] * nb
    for x in nums:
        i = (x - mn) // size
        bmin[i] = min(bmin[i], x)
        bmax[i] = max(bmax[i], x)
    steps = [_step("scatter into buckets", [("bucket min", bmin, None), ("bucket max", bmax, None)],
                 [("width", size, "amber")],
                 eq=f"bucket = (x - {mn}) // {size}; the max gap is >= {size}, a within-bucket gap is <= {size - 1}, so the answer is a seam")]
    gap = 0
    prev = bmax[0]
    for b in range(1, nb):
        if math.isinf(bmin[b]):
            continue
        d = int(bmin[b] - prev)
        gap = max(gap, d)
        steps.append(_step(f"seam before bucket {b}", [("bucket min", bmin, b), ("bucket max", bmax, b)],
                           [("seam", d, "amber"), ("best", gap, "green")],
                           eq=f"min of bucket {b} ({int(bmin[b])}) minus max of the previous non-empty bucket ({int(prev)}) = {d}"))
        prev = bmax[b]
    assert gap == 3
    return _pack("Maximum Gap - bucket (pigeonhole) linear scan",
                 "nums = [3, 6, 9, 1] -> sorted [1, 3, 6, 9]", steps, gap)


def _rank_compress():
    arr = [40, 10, 20, 30]
    uniq = sorted(set(arr))
    rank = {v: i + 1 for i, v in enumerate(uniq)}
    steps = [_step("sorted distinct values", [("uniq", uniq, None)],
                 [("ranks", [rank[v] for v in uniq], "amber")],
                 eq="coordinate compression: rank = 1-based position in the sorted distinct list")]
    res = []
    for idx, v in enumerate(arr):
        res.append(rank[v])
        steps.append(_step(f"rank of {v}", [("arr", arr, idx)],
                           [("value", v, "indigo"), ("rank", rank[v], "green")],
                           eq=f"{v} sits at index {rank[v] - 1} of the distinct list -> rank {rank[v]}"))
    assert res == [4, 1, 2, 3]
    return _pack("Rank Transform - one-line coordinate compression",
                 "arr = [40, 10, 20, 30]", steps, str(res))


def _freq_group():
    nums = [1, 1, 2, 2, 2, 3]
    cnt = Counter(nums)
    vals = sorted(cnt, key=lambda v: (cnt[v], -v))
    res = []
    steps = [_step("count frequencies", [("nums", nums, None)],
                   [("freq", ", ".join(f"{k}:{v}" for k, v in sorted(cnt.items())), "amber")],
                   eq="sort the distinct values by (frequency ascending, value descending)")]
    for v in vals:
        res += [v] * cnt[v]
        steps.append(_step(f"emit value {v}", [("result", res, len(res) - 1)],
                           [("value", v, "indigo"), ("freq", cnt[v], "amber")],
                           eq=f"{v} occurs {cnt[v]} time(s); equal-frequency ties put the larger value later"))
    assert res == [3, 1, 1, 2, 2, 2]
    return _pack("Sort by Increasing Frequency - a sort on a derived key",
                 "nums = [1, 1, 2, 2, 2, 3]", steps, str(res))


def _stick_median():
    p = [2, 3, 1, 5, 2]
    s = sorted(p)
    med = s[len(s) // 2]
    cost = sum(abs(x - med) for x in p)
    steps = [_step("sorted lengths, median = 2", [("p", s, None)],
                   [("median", med, "amber")],
                   eq="the median minimises the sum of absolute deviations (exchange argument)")]
    run = 0
    for i, x in enumerate(s):
        d = abs(x - med)
        run += d
        steps.append(_step(f"cell {i}: {x}", [("p", s, i)],
                           [("deviation", d, "violet"), ("total", run, "green")],
                           eq=f"|{x} - {med}| = {d}; running cost {run}"))
    assert cost == 5
    return _pack("Stick Lengths - gather everyone to the median",
                 "p = [2, 3, 1, 5, 2] -> sorted [1, 2, 2, 3, 5]", steps, cost)


def _cost_median():
    nums = [1, 3, 5, 2]
    cost = [2, 3, 1, 14]
    pairs = sorted(zip(nums, cost))
    total = sum(cost)
    target = None
    steps = [_step("sort pairs by value", [("nums", [str(v) for v, _ in pairs], None),
                                          ("cost", [str(c) for _, c in pairs], None)],
               [("total weight", total, "amber")],
               eq="weighted median: walk left to right until cumulative cost reaches half of " + str(total))]
    cum = 0
    for i, (v, c) in enumerate(pairs):
        cum += c
        crossed = target is None and 2 * cum >= total
        if crossed:
            target = v
        steps.append(_step(f"value {v}", [("nums", [str(x) for x, _ in pairs], i),
                                          ("cost", [str(x) for _, x in pairs], i)],
                          [("cum", cum, "violet"), ("half", total / 2, "amber"),
                           ("target", target if target is not None else "-", "green")],
                          eq=f"cumulative cost {cum} vs half of {total}" + (" -> weighted median here" if crossed else "")))
    res = sum(c * abs(v - target) for v, c in pairs)
    assert res == 8 and target == 2
    return _pack("Minimum Cost to Make Array Equal - weighted median",
                 "nums = [1,3,5,2], cost = [2,3,1,14]", steps, res)


# ---------------------------------------------------------------------------
# Intermediate-rung traces: the middle step between the direct-definition
# brute and the pattern's insight, so every problem shows the full ladder.
# ---------------------------------------------------------------------------


def _ins_sort_trace(src, title_prefix):
    """Insertion sort on ``src``, one frame per pass; returns (sorted, steps)."""
    b = list(src)
    steps = []
    for i in range(1, len(b)):
        j = i
        while j > 0 and b[j - 1] > b[j]:
            b[j - 1], b[j] = b[j], b[j - 1]
            j -= 1
        steps.append(_step(f"{title_prefix} pass {i}", [("b", b, i)],
                           [("sorted prefix", i + 1, "amber")],
                           eq=f"insertion sort: sink b[{i}] into the sorted prefix via adjacent swaps"))
    return b, steps


def _mindiff_naive():
    arr = [4, 2, 1, 3]
    steps = [_step("input (unsorted)", [("b", arr, None)], [("best", "-", "green")],
                   eq="first pay for sortedness with an O(n^2) insertion sort, then run the neighbour scan")]
    b, ins = _ins_sort_trace(arr, "insertion")
    steps.extend(ins)
    best = min(b[i + 1] - b[i] for i in range(len(b) - 1))
    steps.append(_step("neighbour scan", [("b", b, None)],
                       [("best", best, "green")],
                       eq=f"gaps {b[1]-b[0]}, {b[2]-b[1]}, {b[3]-b[2]} -> min {best}"))
    assert b == [1, 2, 3, 4] and best == 1
    return _pack("Minimum Absolute Difference - insertion sort then neighbour scan",
                 "arr = [4, 2, 1, 3]", steps, best)


def _honest_naive():
    src = [3, 1, 2, 6, 4]
    steps = [_step("input strengths", [("b", src, None)], [("best", "-", "green")],
                   eq="same route as Minimum Absolute Difference: pay n^2 for the order, then one linear seam scan")]
    b, ins = _ins_sort_trace(src, "insertion")
    steps.extend(ins)
    best = min(b[i + 1] - b[i] for i in range(len(b) - 1))
    steps.append(_step("min adjacent gap", [("b", b, None)],
                       [("best", best, "green")],
                       eq=f"gaps {b[1]-b[0]}, {b[2]-b[1]}, {b[3]-b[2]}, {b[4]-b[3]} -> best seam {best}"))
    assert b == [1, 2, 3, 4, 6] and best == 1
    return _pack("Honest Coach - insertion sort then neighbour scan",
                 "s = [3, 1, 2, 6, 4]", steps, best)


def _queue_walk():
    """Intermediate rung: shortest-first (ties broken by k descending), drop
    each person into their k-th empty slot.  Because everyone shorter than
    the current person is already placed, they occupy exactly ``k`` of the
    empty slots ahead of us; the ``k``-th empty slot is therefore the unique
    index where ``k`` taller-or-equal people will end up in front.  This is a
    genuinely different algorithm from the optimal form (opposite sort
    direction, different bookkeeping) with the same ``O(n^2)`` naive-slot
    cost."""
    people = [[7, 0], [4, 4], [7, 1], [5, 0], [6, 1], [5, 2]]

    def lab(s):
        return "." if s is None else f"{s[0]},{s[1]}"

    order = sorted(people, key=lambda p: (p[0], -p[1]))
    n = len(order)
    slot = [None] * n
    steps = [_step("sort by (h, -k) - shortest first",
                   [("order", [lab(p) for p in order], None)],
                   [("n", n, "amber")],
                   eq="shortest first, so every already-placed person is no taller than us; ties broken by k descending, because equal-height people count toward each other's k")]
    for h, k in order:
        empties = [j for j in range(n) if slot[j] is None]
        j = empties[k]
        slot[j] = [h, k]
        steps.append(_step(f"place {h},{k} at empty #{k}",
                           [("slot", [lab(s) for s in slot], j)],
                           [("k", k, "amber"), ("index", j, "green")],
                           eq=f"skip {k} empty slots (already promised to shorter people), write into slot {j}"))
    assert slot == [[5, 0], [7, 0], [5, 2], [6, 1], [4, 4], [7, 1]]
    return _pack("Queue Reconstruction - sort short-first, k-th empty slot",
                 "people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]", steps, str(slot))


def _relative_select():
    arr1 = [2, 3, 1, 4, 7]
    arr2 = [2, 1, 4, 3]
    INF = len(arr2)
    pos = {v: i for i, v in enumerate(arr2)}
    left = list(arr1)
    res = []
    steps = [_step("input", [("left", left, None)],
                   [("arr2", arr2, "amber")],
                   eq="selection-sort style: repeatedly pull the smallest remaining under the (rank, value) key")]
    while left:
        pick = min(left, key=lambda x: (pos.get(x, INF), x))
        idx = left.index(pick)
        left.pop(idx)
        res.append(pick)
        steps.append(_step(f"pull {pick}",
                           [("left", left, idx), ("res", res, len(res) - 1)],
                           [("rank", pos.get(pick, INF), "indigo"), ("size res", len(res), "green")],
                           eq=f"minimum of the remaining list under key {(pos.get(pick, INF), pick)}"))
    assert res == [2, 1, 4, 3, 7]
    return _pack("Relative Sort Array - selection sort by (rank, value) key",
                 "arr1 = [2,3,1,4,7], arr2 = [2,1,4,3]", steps, str(res))


def _relative_cmp():
    arr1 = [2, 3, 1, 3, 2, 4, 7]
    arr2 = [2, 1, 4, 3]
    pos = {v: i for i, v in enumerate(arr2)}
    INF = len(arr2)
    keys = [f"({pos.get(x, INF)},{x})" for x in arr1]
    res = sorted(arr1, key=lambda x: (pos.get(x, INF), x))
    steps = [_step("input arr1 and order map", [("arr1", arr1, None)],
                   [("pos", ", ".join(f"{k}:{v}" for k, v in arr2_ranked(arr2)), "amber")],
                   eq="give every value its rank in arr2 (or INF if absent); sentinel sorts unranked to the end")]
    steps.append(_step("keys per element",
                       [("arr1", arr1, None), ("keys", keys, None)],
                       [("sort key", "(rank, value)", "indigo")],
                       eq="rank orders by arr2 first; value breaks ties inside the INF bucket (leftovers ascending)"))
    steps.append(_step("comparison sort output", [("result", res, None)],
                       [("after sort", "-", "green")],
                       eq="Timsort on the tuple keys: [2,2] rank 0, [1] rank 1, [4] rank 2, [3,3] rank 3, [7] rank INF"))
    assert res == [2, 2, 1, 4, 3, 3, 7]
    return _pack("Relative Sort Array - comparison sort by (rank, value) key",
                 "arr1 = [2,3,1,3,2,4,7], arr2 = [2,1,4,3]", steps, str(res))


def arr2_ranked(arr2):
    for i, v in enumerate(arr2):
        yield v, i


def _gap_pair_brute():
    nums = [3, 6, 9, 1]
    steps = [_step("input", [("nums", nums, None)], [("best", "-", "green")],
                   eq="definition: a pair (x, y) is successive in sorted order iff no value lies strictly between them")]
    best = 0
    shown = 0
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i == j:
                continue
            x, y = nums[i], nums[j]
            if x >= y:
                continue
            inside = any(x < z < y for z in nums)
            if inside:
                continue
            d = y - x
            if d > best:
                best = d
            shown += 1
            steps.append(_step(f"successive {x} | {y}", [("nums", nums, i)],
                               [("diff", d, "amber"), ("best", best, "green")],
                               eq=f"no value between {x} and {y}: this pair is adjacent when sorted; gap {d}"))
            if shown >= 3:
                break
        if shown >= 3:
            break
    assert best == 3
    return _pack("Maximum Gap - direct definition (successive pairs)",
                 "nums = [3, 6, 9, 1]", steps, best)


def _gap_sort():
    nums = [3, 6, 9, 1]
    b = sorted(nums)
    steps = [_step("sort", [("b", b, None)], [("best", "-", "green")],
                   eq="the O(n log n) comparison route LC 164 disallows as the final word, but the right oracle for the bucket proof")]
    best = 0
    for i in range(1, len(b)):
        d = b[i] - b[i - 1]
        best = max(best, d)
        steps.append(_step(f"gap at {i - 1}", [("b", b, i)],
                           [("gap", d, "amber"), ("best", best, "green")],
                           eq=f"b[{i}] - b[{i-1}] = {d}; running max {best}"))
    assert best == 3
    return _pack("Maximum Gap - sort then neighbour max scan",
                 "nums = [3, 6, 9, 1] -> sorted [1, 3, 6, 9]", steps, best)


def _rank_bisect():
    arr = [40, 10, 20, 30]
    uniq = sorted(set(arr))
    steps = [_step("sorted distinct", [("uniq", uniq, None)],
                   [("m", len(uniq), "amber")],
                   eq="rank of v = bisect_left(uniq, v) + 1 - a binary search per element instead of a dict hit")]
    res = []
    for idx, v in enumerate(arr):
        r = bisect.bisect_left(uniq, v) + 1
        res.append(r)
        steps.append(_step(f"bisect {v}",
                           [("uniq", uniq, r - 1), ("arr", arr, idx)],
                           [("index", r - 1, "violet"), ("rank", r, "green")],
                           eq=f"binary search finds {v} at position {r-1} of uniq -> rank = {r}"))
    assert res == [4, 1, 2, 3]
    return _pack("Rank Transform - sort distinct then bisect each element",
                 "arr = [40, 10, 20, 30]", steps, str(res))


def _stick_prefix():
    p = [2, 3, 1, 5, 2]
    b = sorted(p)
    n = len(b)
    pref = [0]
    for x in b:
        pref.append(pref[-1] + x)
    steps = [_step("sorted + prefix sums",
                   [("b", b, None), ("pref", pref, None)],
                   [("n", n, "amber")],
                   eq="cost(b[i]) = b[i]*(i+1) - pref[i+1] + (pref[n]-pref[i+1]) - b[i]*(n-i-1): O(1) per candidate target")]
    best = math.inf
    for i in range(n):
        t = b[i]
        lc = t * (i + 1) - pref[i + 1]
        rc = (pref[n] - pref[i + 1]) - t * (n - i - 1)
        cost = lc + rc
        best = min(best, cost)
        steps.append(_step(f"target = {t}",
                           [("b", b, i), ("pref", pref, None)],
                           [("left", lc, "violet"), ("right", rc, "cyan"),
                            ("cost", cost, "amber"), ("best", best, "green")],
                           eq=f"cost({t}) = {lc} + {rc} = {cost}"))
    assert best == 5
    return _pack("Stick Lengths - sort + prefix sums, evaluate cost at every data point",
                 "p = [2,3,1,5,2] -> sorted [1,2,2,3,5]", steps, best)


def _cost_prefix():
    nums = [1, 3, 5, 2]
    cst = [2, 3, 1, 14]
    pairs = sorted(zip(nums, cst))
    n = len(pairs)
    pref_w = [0]
    pref_ws = [0]
    for v, c in pairs:
        pref_w.append(pref_w[-1] + c)
        pref_ws.append(pref_ws[-1] + c * v)
    steps = [_step("sorted pairs + weighted prefix",
                   [("v", [str(v) for v, _ in pairs], None),
                    ("c", [str(c) for _, c in pairs], None),
                    ("pref_w", pref_w, None)],
                   [("total W", pref_w[n], "amber")],
                   eq="cost(v_i) = v_i*pref_w[i+1] - pref_ws[i+1] + (pref_ws[n]-pref_ws[i+1]) - v_i*(pref_w[n]-pref_w[i+1])")]
    best = math.inf
    for i in range(n):
        v = pairs[i][0]
        lc = v * pref_w[i + 1] - pref_ws[i + 1]
        rc = (pref_ws[n] - pref_ws[i + 1]) - v * (pref_w[n] - pref_w[i + 1])
        cost = lc + rc
        best = min(best, cost)
        steps.append(_step(f"target = {v}",
                           [("v", [str(x) for x, _ in pairs], i),
                            ("pref_w", pref_w, None)],
                           [("left", lc, "violet"), ("right", rc, "cyan"),
                            ("cost", cost, "amber"), ("best", best, "green")],
                           eq=f"cost({v}) = {lc} + {rc} = {cost}"))
    assert best == 8
    return _pack("Minimum Cost to Make Array Equal - weighted prefix, evaluate cost at every value",
                 "nums = [1,3,5,2], cost = [2,3,1,14]", steps, best)


# ---------------------------------------------------------------------------
# Brute-rung traces: the direct-definition pass for every problem, so the
# reader can watch the naive pass decide and compare it to the smarter rungs.
# ---------------------------------------------------------------------------


def _mindiff_pair_brute():
    arr = [5, 2, 8, 1]
    steps = [_step("input", [("arr", arr, None)], [("best", "-", "green")],
                   eq="check every unordered pair (i, j), i < j; track the smallest |diff| and every pair at it")]
    best = math.inf
    pairs = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            d = abs(arr[i] - arr[j])
            pair = [min(arr[i], arr[j]), max(arr[i], arr[j])]
            if d < best:
                best, pairs = d, [pair]
            elif d == best:
                pairs.append(pair)
            steps.append(_step(f"pair {arr[i]},{arr[j]}", [("arr", arr, j)],
                               [("|diff|", d, "amber"), ("best", best, "green")],
                               eq=f"|{arr[i]} - {arr[j]}| = {d}" + (" -> new min" if d == best else "")))
    assert best == 1 and pairs == [[1, 2]]
    return _pack("Minimum Absolute Difference - all-pairs scan (brute)",
                 "arr = [5, 2, 8, 1]", steps, str(sorted(pairs)))


def _honest_pair_brute():
    s = [3, 1, 2, 6]
    steps = [_step("input strengths", [("s", s, None)], [("best", "-", "green")],
                   eq="for a two-team split the seam equals the difference of some pair; scan every pair and take the min")]
    best = math.inf
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            d = abs(s[i] - s[j])
            improved = d < best
            if improved:
                best = d
            steps.append(_step(f"pair {s[i]},{s[j]}", [("s", s, j)],
                               [("gap", d, "amber"), ("best", best, "green")],
                               eq=f"|{s[i]} - {s[j]}| = {d}" + (" -> new best seam" if improved else "")))
    assert best == 1
    return _pack("Honest Coach - all-pairs scan (brute)",
                 "s = [3, 1, 2, 6]", steps, best)


def _queue_perm_brute():
    """Brute rung for LC 406: enumerate every permutation of a small valid
    input and keep the first ordering in which each person sees exactly ``k``
    taller-or-equal people ahead."""
    people = [[5, 0], [4, 1], [3, 2]]

    def valid(q):
        for i in range(len(q)):
            h, k = q[i]
            if sum(1 for j in range(i) if q[j][0] >= h) != k:
                return False
        return True

    steps = [_step("input queue candidates",
                   [("people", [f"{a},{b}" for a, b in people], None)],
                   [("n!", math.factorial(len(people)), "amber")],
                   eq="enumerate permutations in lexicographic order; accept the first whose every position has k taller-or-equal people ahead")]
    shown = 0
    for perm in permutations(people):
        cand = [list(p) for p in perm]
        ok = valid(cand)
        shown += 1
        steps.append(_step(f"try #{shown}",
                           [("perm", [f"{a},{b}" for a, b in cand], None)],
                           [("valid", "yes" if ok else "no", "green" if ok else "red")],
                           eq=("every position matches its k - stop here" if ok else
                               "first mismatching position rejects this ordering")))
        if ok:
            return _pack("Queue Reconstruction - permutation search (brute)",
                         "people = [[5,0],[4,1],[3,2]]", steps, str(cand))
    raise AssertionError("no valid permutation - sample input is wrong")


def _rank_count_brute():
    arr = [40, 10, 20, 30]
    uniq = set(arr)
    steps = [_step("distinct values", [("arr", arr, None)],
                   [("|uniq|", len(uniq), "amber")],
                   eq="rank(v) = 1 + |{u in uniq : u < v}| - a fresh count scan for every position")]
    res = []
    for idx, v in enumerate(arr):
        r = 1 + sum(1 for u in uniq if u < v)
        res.append(r)
        steps.append(_step(f"rank of {v}", [("arr", arr, idx)],
                           [("smaller", r - 1, "violet"), ("rank", r, "green")],
                           eq=f"{r-1} distinct value(s) strictly smaller than {v} -> rank {r}"))
    assert res == [4, 1, 2, 3]
    return _pack("Rank Transform - count distinct smaller, per element (brute)",
                 "arr = [40, 10, 20, 30]", steps, str(res))


def _stick_brute():
    p = [2, 3, 1, 5, 2]
    steps = [_step("input lengths", [("p", p, None)], [("best", "-", "green")],
                   eq="try each stick length as the common target; sum the movement cost, keep the minimum")]
    best = math.inf
    best_t = None
    seen = set()
    for i, t in enumerate(p):
        if t in seen:
            continue
        seen.add(t)
        cost = sum(abs(x - t) for x in p)
        if cost < best:
            best, best_t = cost, t
        steps.append(_step(f"target = {t}", [("p", p, i)],
                           [("cost", cost, "amber"), ("best", best, "green")],
                           eq=f"sum |x - {t}| over all sticks = {cost}"))
    assert best == 5 and best_t == 2
    return _pack("Stick Lengths - try every data point as target (brute)",
                 "p = [2, 3, 1, 5, 2]", steps, best)


def _cost_brute():
    nums = [1, 3, 5, 2]
    cst = [2, 3, 1, 14]
    steps = [_step("input nums and cost",
                   [("nums", nums, None), ("cost", cst, None)],
                   [("|distinct|", len(set(nums)), "amber")],
                   eq="try each distinct value as target; cost is the weighted sum of distances")]
    best = math.inf
    seen = set()
    for i, t in enumerate(nums):
        if t in seen:
            continue
        seen.add(t)
        c = sum(cs * abs(x - t) for x, cs in zip(nums, cst))
        if c < best:
            best = c
        steps.append(_step(f"target = {t}",
                           [("nums", nums, i), ("cost", cst, None)],
                           [("cost(t)", c, "amber"), ("best", best, "green")],
                           eq=f"sum cost_i * |nums_i - {t}| = {c}"))
    assert best == 8
    return _pack("Minimum Cost to Make Array Equal - try every value as target (brute)",
                 "nums = [1,3,5,2], cost = [2,3,1,14]", steps, best)


# ---------------------------------------------------------------------------
# CF 922D Robot Vacuum Cleaner (P2b) and CF 1490F Equalize the Array (P4b).
# ---------------------------------------------------------------------------


def _sh(t):
    """(number of 's', number of 'h', internal 'sh' pairs) of one string."""
    s_cnt = h_cnt = inside = 0
    for ch in t:
        if ch == "s":
            s_cnt += 1
        else:
            h_cnt += 1
            inside += s_cnt
    return s_cnt, h_cnt, inside


def _noise(text):
    return _sh(text)[2]


def _robot_brute():
    ts = ["ssh", "hs", "s", "hhhs"]
    steps = [_step("input strings", [("t", ts, None)], [("orders", math.factorial(len(ts)), "amber")],
                   eq="try every concatenation order; count the 'sh' subsequences of each")]
    best, best_p = -1, None
    shown = 0
    for perm in permutations(ts):
        v = _noise("".join(perm))
        if v > best:
            best, best_p = v, list(perm)
            shown += 1
            steps.append(_step(f"order {shown}: new best", [("order", list(perm), None)],
                               [("noise", v, "amber"), ("best", best, "green")],
                               eq=f"{''.join(perm)} has {v} 'sh' pairs"))
    assert best == 18 and len(steps) <= 8
    return _pack("Robot Vacuum Cleaner - try every order (brute)",
                 "t = [ssh, hs, s, hhhs]", steps, best)


def _robot_cmp():
    ts = ["ssh", "hs", "s", "hhhs"]
    cnt = {t: _sh(t) for t in ts}

    def cmp(a, b):
        x = cnt[a][0] * cnt[b][1]
        y = cnt[b][0] * cnt[a][1]
        return -1 if x > y else (1 if x < y else 0)

    order = sorted(ts, key=cmp_to_key(cmp))
    steps = [_step("count s and h per string", [("t", ts, None), ("s,h", [f"{cnt[t][0]},{cnt[t][1]}" for t in ts], None)],
                   [("rule", "a first iff s(a)h(b) > s(b)h(a)", "indigo")],
                   eq="only the cross pairs between two neighbours change when they swap")]
    steps.append(_step("sort with the pairwise rule", [("order", order, None),
                                                      ("s,h", [f"{cnt[t][0]},{cnt[t][1]}" for t in order], None)],
                       [("first", order[0], "green")],
                       eq="s has no h (never loses), hhhs has many h and one s (goes last)"))
    seen = total = 0
    for i, t in enumerate(order):
        sc, hc, inside = cnt[t]
        cross = seen * hc
        total += cross + inside
        steps.append(_step(f"append {t}", [("order", order, i)],
                           [("s so far", seen, "violet"), ("cross", cross, "amber"),
                            ("inside", inside, "cyan"), ("noise", total, "green")],
                           eq=f"{seen} earlier s x {hc} h in '{t}' = {cross}, plus {inside} inside"))
        seen += sc
    assert total == 18 and order == ["s", "ssh", "hs", "hhhs"]
    return _pack("Robot Vacuum Cleaner - sort by the exchange rule, count while appending",
                 "t = [ssh, hs, s, hhhs]", steps, total)


def _equal_brute():
    a = [1, 2, 3, 3, 3, 2, 6, 6]
    freq = Counter(a)
    n = len(a)
    steps = [_step("count each value", [("a", a, None)],
                   [("freq", ", ".join(f"{k}:{v}" for k, v in sorted(freq.items())), "amber")],
                   eq="for every target count C: values with fewer than C copies go, the rest are trimmed to C")]
    best = math.inf
    for c in range(1, 5):
        cost = sum(f if f < c else f - c for f in freq.values())
        best = min(best, cost)
        steps.append(_step(f"C = {c}", [("a", a, None)],
                           [("deletions", cost, "amber"), ("best", best, "green")],
                           eq=" + ".join(str(f if f < c else f - c) for f in freq.values()) + f" = {cost}"))
    assert best == 2 and n == 8
    return _pack("Equalize the Array - try every target count C (brute)",
                 "a = [1, 2, 3, 3, 3, 2, 6, 6]", steps, best)


def _equal_sorted():
    a = [1, 2, 3, 3, 3, 2, 6, 6]
    n = len(a)
    f = sorted(Counter(a).values())
    m = len(f)
    steps = [_step("sort the frequencies", [("freq", f, None)],
                   [("n", n, "amber"), ("m", m, "violet")],
                   eq="after sorting, the values with count >= f[i] are exactly f[i..m-1]")]
    best_keep = 0
    for i, c in enumerate(f):
        keep = c * (m - i)
        best_keep = max(best_keep, keep)
        steps.append(_step(f"C = f[{i}] = {c}", [("freq", f, i)],
                           [("kept", keep, "cyan"), ("deleted", n - keep, "amber"),
                            ("best", n - best_keep, "green")],
                           eq=f"{m - i} values keep {c} copies each: keep {keep}, delete {n} - {keep} = {n - keep}"))
    assert n - best_keep == 2
    return _pack("Equalize the Array - sorted frequencies, one pass",
                 "a = [1, 2, 3, 3, 3, 2, 6, 6]", steps, n - best_keep)


_ALL = None


def get(problem, form):
    """Return ``(title, input line, steps, final)`` for one solution form."""
    global _ALL
    if _ALL is None:
        _ALL = {
            # P1a - LC 1200 Minimum Absolute Difference: pair scan -> insertion sort + scan -> sort + scan
            ("mindiff", "brute"): _mindiff_pair_brute(),
            ("mindiff", "naive"): _mindiff_naive(),
            ("mindiff", "sort"): _mindiff_sort(),
            # P1b - CF 1360B Honest Coach: pair scan -> insertion sort + scan -> sort + scan
            ("honest", "brute"): _honest_pair_brute(),
            ("honest", "naive"): _honest_naive(),
            ("honest", "sort"): _honest_sort(),
            # P2 - Largest Number (solved in ch. 3) and Queue Reconstruction
            ("largest", "cmp"): _largest_cmp(),
            ("queue", "perm"): _queue_perm_brute(),
            ("queue", "walk"): _queue_walk(),
            ("queue", "greedy"): _queue_greedy(),
            # P3a - LC 1122 Relative Sort Array: selection sort -> key sort -> counting sort
            ("relative", "select"): _relative_select(),
            ("relative", "cmp"): _relative_cmp(),
            ("relative", "count"): _relative_count(),
            # P3b - LC 164 Maximum Gap: all-pairs successive -> sort + scan -> bucket + pigeonhole
            ("gap", "pair"): _gap_pair_brute(),
            ("gap", "sort"): _gap_sort(),
            ("gap", "bucket"): _gap_bucket(),
            # P4 - Rank Transform: count distinct smaller -> bisect -> dict compression
            ("rank", "brute"): _rank_count_brute(),
            ("rank", "bisect"): _rank_bisect(),
            ("rank", "compress"): _rank_compress(),
            ("freq", "group"): _freq_group(),
            # P2b - CF 922D Robot Vacuum Cleaner: every order -> pairwise exchange comparator
            ("robot", "brute"): _robot_brute(),
            ("robot", "cmp"): _robot_cmp(),
            # P4b - CF 1490F Equalize the Array: every C -> sorted frequencies
            ("equal", "brute"): _equal_brute(),
            ("equal", "sorted"): _equal_sorted(),
            # P5 - Stick Lengths and LC 2448: try-every-target -> sort + prefix sums -> (weighted) median
            ("stick", "brute"): _stick_brute(),
            ("stick", "prefix"): _stick_prefix(),
            ("stick", "median"): _stick_median(),
            ("cost", "brute"): _cost_brute(),
            ("cost", "prefix"): _cost_prefix(),
            ("cost", "median"): _cost_median(),
        }
    return _ALL[(problem, form)]
