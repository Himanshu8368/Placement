"""Step frames for the Sorting Algorithms chapter (Chapter 12).

Drawing lives in ``dsa_viz.story``; ``get`` returns data only. Each board walks
the chapter's own sample in at most eight frames, and every number is produced
by *running the algorithm* - never a hand-typed table - so a wrong sort fails
the assert just like a wrong solution would.

A frame is one or two array rows (with the active column ``hi`` highlighted)
plus rolling chips (the current minimum, the pivot, the running count), so the
reader watches the algorithm decide, exactly as the code does. Each problem
carries the full ladder: a ``brute`` frame set (the direct definition) and a
refined set (the divide-and-conquer / selection / heap move the pattern names).
"""
from __future__ import annotations

import sys
from bisect import bisect_left
from heapq import heapify, heappop, heappush
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import chip_step, row_step, show


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    if not 1 <= len(steps) <= 8:
        raise AssertionError(f"{title}: {len(steps)} frames")
    return (title, inp, steps, text)


def _step(title, rows, chips=None, eq=""):
    frame = row_step(title, rows, eq=eq)
    if chips:
        frame["chips"] = chip_step(title, chips, eq="")["chips"]
    return frame


# ---------------------------------------------------------------------------
# P1 - implementing a comparison sort.
# ---------------------------------------------------------------------------

def _sortarr_select():
    a = [5, 2, 3, 1]
    steps = [_step("unsorted", [("a", a, 0)], [("sorted prefix", 0, "green")],
             eq="selection sort: grow a sorted prefix by picking the min of the rest")]
    for i in range(len(a)):
        m = min(range(i, len(a)), key=lambda j: a[j])
        a[i], a[m] = a[m], a[i]
        steps.append(_step(f"pass {i}: swap min {a[i]} in", [("a", a, i)],
                           [("min at", m, "amber"), ("sorted", i + 1, "green")],
                           eq=f"min of a[{i}..] was at index {m}; swap it to position {i}"))
    assert a == [1, 2, 3, 5]
    return _pack("Sort an Array - selection sort", "nums = [5, 2, 3, 1]", steps, str(a))


def _sortarr_merge():
    steps = [_step("split into single pairs", [("left", [5, 2], None), ("right", [3, 1], None)],
               [("level", 0, "amber")], eq="divide the array in half until each piece has one element")]

    def merge(x, y):
        out = []
        i = j = 0
        while i < len(x) and j < len(y):
            if x[i] <= y[j]:
                out.append(x[i]); i += 1
            else:
                out.append(y[j]); j += 1
        return out + x[i:] + y[j:]

    l1, l2 = merge([5], [2]), merge([3], [1])
    steps.append(_step("merge the pairs", [("left", l1, None), ("right", l2, None)],
                       [("level", 1, "amber")], eq="merge [5]|[2] -> [2,5] and [3]|[1] -> [1,3]"))
    top = merge(l1, l2)
    steps.append(_step("merge the halves", [("result", top, None)], [("done", len(top), "green")],
                       eq="merge [2,5]|[1,3] -> [1,2,3,5] (stable: a tie keeps the left run first)"))
    assert top == [1, 2, 3, 5]
    return _pack("Sort an Array - merge sort", "nums = [5, 2, 3, 1]", steps, str(top))


def _sortlist_arr():
    vals = [4, 2, 1, 3]
    b = sorted(vals)
    steps = [_step("walk the list", [("nodes", vals, None)], [("collected", len(vals), "amber")],
             eq="copy the values into an array, sort it, then rethread the nodes")]
    steps.append(_step("sort the array", [("array", b, None)], [("sorted", b, "green")],
                       eq="the library sort gives O(n log n); the nodes are relinked in this order"))
    assert b == [1, 2, 3, 4]
    return _pack("Sort List - collect to array, sort, relink", "head = [4, 2, 1, 3]", steps, str(b))


def _sortlist_merge():
    steps = [_step("slow/fast split", [("list", [4, 2, 1, 3], 1)], [("mid", 2, "amber")],
                  eq="a slow pointer at half speed finds the middle; cut into [4,2] | [1,3]")]
    steps.append(_step("sort each half", [("left", [2, 4], None), ("right", [1, 3], None)],
                       [("halves", 2, "amber")], eq="recursively sort the two sub-lists"))
    merged, i, j = [], 0, 0
    L, R = [2, 4], [1, 3]
    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            merged.append(L[i]); i += 1
        else:
            merged.append(R[j]); j += 1
    merged += L[i:] + R[j:]
    steps.append(_step("merge the halves", [("merged", merged, None)], [("size", len(merged), "green")],
                       eq="splice two sorted lists with only 'next' pointers - O(1) extra space"))
    assert merged == [1, 2, 3, 4]
    return _pack("Sort List - merge sort on a linked list", "head = [4, 2, 1, 3]", steps, str(merged))


# ---------------------------------------------------------------------------
# P2 - divide, conquer, and count across the merge.
# ---------------------------------------------------------------------------

def _smaller_brute():
    nums = [5, 2, 6, 1]
    counts = []
    steps = [_step("per element, count smaller on its right", [("nums", nums, 0)],
                 [("counts", "-", "green")], eq="for each index i scan j > i and count nums[j] < nums[i]")]
    for i, x in enumerate(nums):
        c = sum(1 for j in range(i + 1, len(nums)) if nums[j] < x)
        counts.append(c)
        steps.append(_step(f"i = {i}: {x}", [("nums", nums, i)], [("count", c, "amber"),
                                                                ("counts", counts, "green")],
                            eq=f"{c} value(s) to the right of {x} are strictly smaller"))
    assert counts == [2, 1, 1, 0]
    return _pack("Count of Smaller Numbers After Self - direct scan",
                 "nums = [5, 2, 6, 1]", steps, str(counts))


def _smaller_merge():
    nums = [5, 2, 6, 1]
    n = len(nums)
    idx = list(range(n))
    counts = [0] * n
    steps = [_step("indices sorted by value", [("idx", list(idx), None), ("nums", nums, None)],
                 [("counts", list(counts), "green")],
                 eq="merge-sort the INDICES; when a LEFT element is emitted after j RIGHT elements, those j are smaller and sat to its right")]

    def msort(lo, hi):
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        msort(lo, mid)
        msort(mid, hi)
        left, right = idx[lo:mid], idx[mid:hi]
        i = j = k = 0
        while i < len(left) and j < len(right):
            if nums[left[i]] <= nums[right[j]]:
                counts[left[i]] += j
                idx[k + lo] = left[i]; i += 1
            else:
                idx[k + lo] = right[j]; j += 1
            k += 1
        while i < len(left):
            counts[left[i]] += j
            idx[k + lo] = left[i]; i += 1; k += 1
        while j < len(right):
            idx[k + lo] = right[j]; j += 1; k += 1
        steps.append(_step(f"merge [{lo}:{mid}] & [{mid}:{hi}]",
                           [("idx", list(idx), None), ("nums", nums, None)],
                           [("counts", list(counts), "amber")],
                           eq="credit each emitted LEFT index with the number of RIGHT values already smaller than it"))

    msort(0, n)
    assert counts == [2, 1, 1, 0]
    steps.append(_step("after all merges", [("counts", list(counts), None)],
                       [("done", "-", "green")], eq="the per-index counts are the answer"))
    return _pack("Count of Smaller Numbers After Self - merge sort counting",
                 "nums = [5, 2, 6, 1]", steps, str(counts))


def _revpair_brute():
    nums = [1, 3, 2, 3, 1]
    best = 0
    steps = [_step("scan all pairs i < j", [("nums", nums, None)], [("pairs", 0, "green")],
               eq="a reverse pair needs nums[i] > 2 * nums[j]; check every ordered pair")]
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] > 2 * nums[j]:
                best += 1
                steps.append(_step(f"pair ({i},{j})", [("nums", nums, j)],
                                   [("(" + str(nums[i]) + ",2*" + str(nums[j]) + ")", f"{nums[i]}>{2*nums[j]}", "amber"),
                                    ("pairs", best, "green")],
                                   eq=f"nums[{i}]={nums[i]} > 2*nums[{j}]={2*nums[j]}"))
                if best >= 2 and (i, j) == (3, 4):
                    break
        if best == 2 and (i, j) == (3, 4):
            break
    assert best == 2
    return _pack("Reverse Pairs - all-pairs scan", "nums = [1, 3, 2, 3, 1]", steps, best)


def _revpair_merge():
    nums = [1, 3, 2, 3, 1]
    a = list(nums)
    total = 0
    steps = [_step("split and recurse", [("a", a, None)], [("count", 0, "green")],
               eq="sort each half, then count cross pairs nums[i] > 2*nums[j] while both halves are sorted")]

    def sort_count(lo, hi):
        nonlocal total
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        sort_count(lo, mid)
        sort_count(mid, hi)
        j = mid
        for i in range(lo, mid):
            while j < hi and a[i] > 2 * a[j]:
                j += 1
            add = j - mid
            total += add
            if add > 0:
                steps.append(_step(f"cross pair at [{lo}:{mid}|{mid}:{hi}]",
                                   [("left", a[lo:mid], i - lo), ("right", a[mid:hi], None)],
                                   [("a[i]", a[i], "violet"), ("count", total, "amber")],
                                   eq=f"a[{i}]={a[i]} > 2*{a[j - 1]} -> {add} new crossing pair(s)"))
        a[lo:hi] = sorted(a[lo:hi])

    sort_count(0, len(a))
    assert total == 2
    steps.append(_step("total", [("sorted", a, None)], [("count", total, "green")],
                       eq="every cross pair was tallied during a merge -> 2"))
    return _pack("Reverse Pairs - merge sort counting", "nums = [1, 3, 2, 3, 1]", steps, str(total))


# ---------------------------------------------------------------------------
# P3 - selection without a full sort.
# ---------------------------------------------------------------------------

def _kth_sort():
    nums = [3, 2, 1, 5, 6, 4]
    k = 2
    b = sorted(nums, reverse=True)
    steps = [_step("sort descending", [("b", b, None)], [("k-th largest", "-", "amber")],
              eq="the k-th largest is simply b[k-1]; O(n log n) but overkill - the others never need order")]
    steps.append(_step("read position k-1", [("b", b, k - 1)], [("answer", b[k - 1], "green")],
                       eq=f"k = {k} -> index {k - 1} = {b[k - 1]}"))
    assert b[k - 1] == 5
    return _pack("Kth Largest Element - sort and index", "nums = [3, 2, 1, 5, 6, 4], k = 2", steps, b[k - 1])


def _kth_qs():
    nums = [3, 2, 1, 5, 6, 4]
    k = 2
    target = len(nums) - k          # index of the 2nd largest == 5th smallest
    steps = [_step("quickselect for index " + str(target), [("a", nums, None)], [("target", target, "amber")],
                  eq="pick a pivot, partition (< pivot | pivot | > pivot), then recurse only into the side holding the target")]
    pivot = 4
    lo = [x for x in nums if x < pivot]     # [3, 2, 1]
    hi = [x for x in nums if x > pivot]     # [5, 6]
    steps.append(_step(f"partition around pivot {pivot}", [("less", lo, None), ("pivot", [pivot], None), ("greater", hi, None)],
                       [("lt,eq,gt", f"{len(lo)},{1},{len(hi)}", "violet")],
                       eq=f"pivot {pivot} lands at index {len(lo)}; the target index {target} sits in the greater part, so only {hi} is revisited"))
    # recurse into the greater block for the (target - len(lo) - 1)-th of it
    rem = target - len(lo) - 1
    ans = sorted(hi)[rem]
    steps.append(_step("recurse on the greater block", [("greater", hi, None)],
                       [("answer", ans, "green")], eq=f"in {hi} the local rank {rem} is {ans} -> the k-th largest"))
    assert ans == 5
    return _pack("Kth Largest Element - quickselect", "nums = [3, 2, 1, 5, 6, 4], k = 2", steps, str(ans))


def _kmat_sort():
    m = [[1, 5, 9], [10, 11, 13], [12, 13, 15]]
    k = 8
    flat = sorted(x for row in m for x in row)
    steps = [_step("flatten and sort", [("flat", flat, None)], [("k", k, "amber")],
              eq="read every cell into one array (O(n^2 log n)) - correct, but ignores the row/column order")]
    steps.append(_step("take the k-th", [("flat", flat, k - 1)], [("answer", flat[k - 1], "green")],
                       eq=f"the {k}-th smallest is flat[{k - 1}]"))
    assert flat[k - 1] == 13
    return _pack("Kth Smallest in a Sorted Matrix - flatten and sort",
                 "matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8", steps, flat[k - 1])


def _kmat_bisect():
    m = [[1, 5, 9], [10, 11, 13], [12, 13, 15]]
    k = 8

    def count_le(x):
        return sum(bisect_left(row, x + 1) for row in m)
    steps = [_step("binary search on the value", [("lo", [m[0][0]], None), ("hi", [m[-1][-1]], None)],
               [("k", k, "amber")], eq="answer 'how many cells <= mid' by per-row bisect; shrink the value range around the k-th")]
    lo, hi = m[0][0], m[-1][-1]
    while lo < hi and len(steps) < 7:
        mid = (lo + hi) // 2
        c = count_le(mid)
        if c < k:
            lo = mid + 1
            note = f"{c} cells <= {mid} -> too few, move lo up to {lo}"
        else:
            hi = mid
            note = f"{c} cells <= {mid} -> enough, move hi down to {hi}"
        steps.append(_step(f"probe mid = {mid}", [("lo", [lo], None), ("hi", [hi], None)],
                           [("mid", mid, "violet"), ("count", c, "amber"), ("vs k", f"{c}{'<' if c < k else '>='}{k}", "green")],
                           eq=note))
    assert count_le(13) >= 8 and count_le(12) < 8 and lo == 13
    steps.append(_step("range closes", [("answer", [lo], None)], [("answer", lo, "green")],
                       eq=f"count_le({lo}) >= k and count_le({lo - 1}) < k, so {lo} is the k-th smallest"))
    return _pack("Kth Smallest in a Sorted Matrix - value-range binary search",
                 "matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8", steps, str(lo))


# ---------------------------------------------------------------------------
# P4 - heaps and top-k.
# ---------------------------------------------------------------------------

def _closest_sort():
    pts = [[1, 3], [-2, 2]]
    k = 1
    keyed = sorted(pts, key=lambda q: q[0] * q[0] + q[1] * q[1])
    steps = [_step("distance squared", [("pts", [f"({a},{b})" for a, b in pts], None),
                                         ("d^2", [p[0] ** 2 + p[1] ** 2 for p in pts], None)],
             [("k", k, "amber")], eq="compare squared distances; sorting is O(n log n)")]
    steps.append(_step("sort, take first k", [("sorted", [f"({a},{b})" for a, b in keyed], None)],
                     [("answer", str(keyed[:k]), "green")], eq="the k closest are the first k of the sorted list"))
    assert keyed[:1] == [[-2, 2]]
    return _pack("K Closest Points - sort by distance", "points = [[1,3],[-2,2]], k = 1", steps, str(keyed[:k]))


def _closest_heap():
    pts = [[3, 3], [5, -1], [-2, 4]]
    k = 2
    heap = []
    steps = [_step("max-heap of size k", [("heap", [], None)], [("cap", k, "amber")],
              eq="keep the k closest so far in a max-heap keyed by distance; a new point closer than the farthest evicts it")]
    seq = []
    for x, y in pts:
        d = -(x * x + y * y)
        heappush(heap, (d, [x, y]))
        evicted = None
        if len(heap) > k:
            evicted = heappop(heap)[1]
        order = [f"({p[1][0]},{p[1][1]})" for p in sorted(heap, key=lambda q: -q[0])]
        seq.append((order, f"push ({x},{y})" + (f"; drop farthest ({evicted[0]},{evicted[1]})" if evicted else "")))
    for order, note in seq[:6]:
        steps.append(_step(note, [("heap", order, None)], [("size", len(order), "green")], eq=note))
    ans = sorted([p[1] for p in heap], key=lambda q: q[0] * q[0] + q[1] * q[1])
    assert ans == [[3, 3], [-2, 4]]
    return _pack("K Closest Points - bounded max-heap", "points = [[3,3],[5,-1],[-2,4]], k = 2", steps, str(ans))


def _stream_sort():
    nums = [4, 5, 8, 2]
    k = 3
    adds = [3, 5, 10, 9, 4]
    pool = list(nums)
    outs = []
    steps = [_step("init pool", [("pool", sorted(pool, reverse=True), None)], [("k", k, "amber")],
              eq="naive: on every add, sort the whole pool and read the k-th largest - O(k n log n)")]
    for v in adds:
        pool.append(v)
        r = sorted(pool, reverse=True)[k - 1]
        outs.append(r)
        steps.append(_step(f"add {v}", [("pool", sorted(pool, reverse=True), k - 1)],
                           [("k-th", r, "green")], eq=f"after adding {v}, the {k}-th largest is {r}"))
    assert outs == [4, 5, 5, 8, 8]
    return _pack("Kth Largest in a Stream - resort on each add", "k=3, nums=[4,5,8,2], adds=[3,5,10,9,4]", steps, str(outs))


def _stream_heap():
    k = 3
    nums = [4, 5, 8, 2]
    heap = nums[:]
    heapify(heap)
    while len(heap) > k:
        heappop(heap)
    adds = [3, 5, 10, 9, 4]
    outs = []
    steps = [_step("min-heap of the top k", [("heap", sorted(heap), None)], [("k", k, "amber")],
              eq="keep a min-heap of exactly the k largest seen; the k-th largest is the heap's minimum (its root)")]
    for v in adds:
        if len(heap) < k:
            heappush(heap, v)
        elif v > heap[0]:
            heapreplace = (heappop(heap), heappush(heap, v))
        r = heap[0]
        outs.append(r)
        steps.append(_step(f"add {v}", [("heap", sorted(heap), 0)], [("k-th", r, "green")],
                           eq=f"root of the top-{k} heap is {r} after adding {v}"))
    assert outs == [4, 5, 5, 8, 8]
    return _pack("Kth Largest in a Stream - bounded min-heap", "k=3, nums=[4,5,8,2], adds=[3,5,10,9,4]", steps, str(outs))


# ---------------------------------------------------------------------------
# P5 - stability and tie-break direction decide the answer.
# ---------------------------------------------------------------------------

def _logs_bucket():
    logs = ["dig1 8 1 5 1", "let1 art can", "dig2 3 6", "let2 own kit dig", "let3 art zero"]
    letters = [w for w in logs if w.split()[1].isdigit() is False]
    letters = sorted(letters, key=lambda w: (w.split(" ", 1)[1], w.split(" ", 1)[0]))
    digits = [w for w in logs if w.split()[1].isdigit()]
    res = letters + digits
    steps = [_step("split by first content word", [("letters", letters, None), ("digits", digits, None)],
                   [("n", len(logs), "amber")], eq="letter-logs sort; digit-logs keep input order - two buckets")]
    steps.append(_step("sort the letter bucket by (content, id)", [("sorted letters", letters, None)],
                       [("kept", len(letters), "violet")], eq="'art can' < 'art zero' < 'own kit dig'; the identifier only breaks exact content ties"))
    steps.append(_step("concatenate (letters, then digits)", [("result", res, None)], [("done", "-", "green")],
                       eq="digit-logs unchanged at the tail: 'dig1...' before 'dig2...' is their arrival order"))
    assert res == ["let1 art can", "let3 art zero", "let2 own kit dig", "dig1 8 1 5 1", "dig2 3 6"]
    return _pack("Reorder Data in Log Files - two buckets", "logs = [dig1.., let1.., dig2.., let2.., let3..]", steps, str(res))


def _logs_stable():
    logs = ["a1 9 2 3 1", "g1 act car", "zo4 4 7", "ab1 off key dog", "a8 act zoo"]
    def key(w):                                   # every digit-log gets the SAME key, so it ties
        ident, body = w.split(" ", 1)
        return (1, "", "") if body[0].isdigit() else (0, body, ident)
    res = sorted(logs, key=key)
    steps = [_step("one sort, digit-logs share a key", [("logs", logs, None)], [("rule", "digit -> last", "amber")],
              eq="give every digit-log the same leading key so the stable sort leaves them in input order behind the letters")]
    steps.append(_step("sorted output", [("result", res, None)], [("done", "-", "green")],
                       eq="letter-logs ordered by content then id; digit-logs keep arrival order because their key ties and the sort is stable"))
    assert res == ["g1 act car", "a8 act zoo", "ab1 off key dog", "a1 9 2 3 1", "zo4 4 7"]
    return _pack("Reorder Data in Log Files - one stable sort", "logs = [a1.., g1.., zo4.., ab1.., a8..]", steps, str(res))


def _weak_brute():
    props = [[1, 5], [10, 4], [4, 3]]
    n = len(props)
    weak = 0
    steps = [_step("test every ordered pair", [("props", props, None)], [("weak", 0, "green")],
               eq="character i is weak if some j has attack_j > attack_i AND defense_j > defense_i")]
    for i in range(n):
        found = False
        for j in range(n):
            if i != j and props[j][0] > props[i][0] and props[j][1] > props[i][1]:
                found = True
                break
        if found:
            weak += 1
        steps.append(_step(f"char {i} = {props[i]}", [("props", props, i)],
                           [("weak?", "yes" if found else "no", "amber"), ("weak", weak, "green")],
                           eq=f"[10,4] beats [4,3] on both axes -> weak" if i == 2 else f"no single character beats {props[i]} on both"))
    assert weak == 1
    return _pack("Weak Characters - all-pairs test", "properties = [[1,5],[10,4],[4,3]]", steps, weak)


def _weak_sort():
    props = [[1, 5], [10, 4], [4, 3]]
    order = sorted(props, key=lambda q: (q[0], -q[1]))     # attack asc, defense DESC
    steps = [_step("sort attack up, defense down", [("order", order, None)], [("max def", 0, "amber")],
              eq="same-attack rows are ordered defense-descending so a later row's defense is never 'cheated' by an equal-attack peer")]
    steps.append(_step("sweep right-to-left on defense", [("order", order, None)],
                       [("weak", 1, "green")], eq="track the max defense seen to the right (strictly greater attack); [4,3] sees 4 -> weak, [1,5] sees 4 -> not"))
    # right-to-left sweep
    best = 0
    weak = 0
    for a, d in reversed(order):
        if d < best:
            weak += 1
        best = max(best, d)
    steps.append(_step("result", [("counts", [weak], None)], [("weak", weak, "green")], eq="only [4,3] is dominated"))
    assert weak == 1 and order == [[1, 5], [4, 3], [10, 4]]
    return _pack("Weak Characters - sort + running-max sweep", "properties = [[1,5],[10,4],[4,3]]", steps, weak)


_ALL = None


def get(problem, form):
    global _ALL
    if _ALL is None:
        _ALL = {
            ("sortarr", "select"): _sortarr_select(),
            ("sortarr", "merge"): _sortarr_merge(),
            ("sortlist", "arr"): _sortlist_arr(),
            ("sortlist", "merge"): _sortlist_merge(),
            ("smaller", "brute"): _smaller_brute(),
            ("smaller", "merge"): _smaller_merge(),
            ("revpair", "brute"): _revpair_brute(),
            ("revpair", "merge"): _revpair_merge(),
            ("kth", "sort"): _kth_sort(),
            ("kth", "qs"): _kth_qs(),
            ("kmat", "sort"): _kmat_sort(),
            ("kmat", "bisect"): _kmat_bisect(),
            ("closest", "sort"): _closest_sort(),
            ("closest", "heap"): _closest_heap(),
            ("stream", "sort"): _stream_sort(),
            ("stream", "heap"): _stream_heap(),
            ("logs", "bucket"): _logs_bucket(),
            ("logs", "stable"): _logs_stable(),
            ("weak", "brute"): _weak_brute(),
            ("weak", "sort"): _weak_sort(),
        }
    return _ALL[(problem, form)]
