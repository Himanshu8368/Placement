"""Step frames for the binary-search-as-a-tool chapter.

Drawing lives in ``dsa_viz.story``. ``get`` returns data only.
Each story walks the chapter's own sample, at most eight frames, and the
numbers come from running the algorithm - never from a hand-typed table.
A binary-search frame shows the array (usually sorted, sometimes a prefix
sum) with the probed index highlighted plus the rolling chips (lo, hi, mid,
counts, best), so the reader watches a *predicate* decide which half dies.

Every problem carries the full ladder: a ``brute`` frame set (the direct
definition), an ``intermediate`` frame set (a linear scan, a two-pointer
walk, or sort-plus-bisect that improves on the brute but has not yet earned
the pattern's *insight*), and an ``optimal`` frame set (the rung the pattern
names). ``get`` returns whichever frame set the caller asks for.
"""
from __future__ import annotations

import bisect
import itertools
import math
import sys
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


# ---------------------------------------------------------------------------
# P1a - LC 2529 Maximum Count of Positive Integer and Negative Integer
# ---------------------------------------------------------------------------

def _signs_scan_brute():
    nums = [-4, -3, -1, -1, 0, 2, 3, 5]
    steps = []
    neg = pos = 0
    for i, x in enumerate(nums):
        if x < 0:
            neg += 1
        elif x > 0:
            pos += 1
        steps.append(_step(f"x = {x}", [("nums", nums, i)],
                           [("neg", neg, "red"), ("pos", pos, "amber"),
                            ("max", max(neg, pos), "green")],
                           eq=(f"{x} < 0 -> neg := {neg}" if x < 0 else
                               f"{x} > 0 -> pos := {pos}" if x > 0 else
                               f"{x} = 0 -> counted as neither")))
    assert (neg, pos) == (4, 3)
    return _pack("Maximum Count of Positive and Negative - two linear counts (brute)",
                 "nums = [-4, -3, -1, -1, 0, 2, 3, 5]", steps, max(neg, pos))


def _signs_count_naive():
    nums = [-4, -3, -1, -1, 0, 2, 3, 5]
    steps = [_step("sorted: the signs run", [("nums", nums, None)],
                  [("neg", "-", "red"), ("pos", "-", "amber")],
                  eq="negatives are a prefix, positives a suffix; the zeros sit between")]
    neg = bisect.bisect_left(nums, 0)
    first_pos = bisect.bisect_left(nums, 1)
    pos = len(nums) - first_pos
    assert (neg, pos) == (4, 3)
    steps.append(_step(f"count negatives = {neg}", [("nums", nums, neg)],
                       [("neg", neg, "red"), ("pos", "-", "amber")],
                       eq=f"everything left of index {neg} is < 0"))
    steps.append(_step(f"count positives = {pos}", [("nums", nums, first_pos)],
                       [("neg", neg, "red"), ("pos", pos, "amber")],
                       eq=f"from index {first_pos} on, everything is > 0"))
    steps.append(_step("take the larger", [("nums", nums, None)],
                       [("neg", neg, "red"), ("pos", pos, "amber"),
                        ("answer", max(neg, pos), "green")],
                       eq=f"max({neg}, {pos}) = {max(neg, pos)}"))
    return _pack("Maximum Count of Positive and Negative - count each run (intermediate)",
                 "nums = [-4, -3, -1, -1, 0, 2, 3, 5]", steps, max(neg, pos))


def _signs_bisect_opt():
    nums = [-4, -3, -1, -1, 0, 2, 3, 5]
    steps = [_step("sorted nums", [("nums", nums, None)],
               [("lo", 0, "indigo"), ("hi", len(nums), "violet")],
               eq="P(x) = 'x >= 0' is false,true,true,...: one bisect_left finds its first true")]
    a = bisect.bisect_left(nums, 0)
    steps.append(_step("bisect_left(nums, 0)", [("nums", nums, a)],
                       [("first >= 0", a, "indigo"), ("neg", a, "red")],
                       eq=f"first index with nums[i] >= 0 is {a}, so exactly {a} negatives"))
    steps.append(_step("P(x) = 'x > 0' shifted", [("nums", nums, a)],
                       [("lo", a, "indigo"), ("hi", len(nums), "violet")],
                       eq="the second predicate 'x > 0' is the same shape, just one value later"))
    b = bisect.bisect_left(nums, 1)
    steps.append(_step("bisect_left(nums, 1)", [("nums", nums, b)],
                       [("first >= 1", b, "indigo"), ("pos", len(nums) - b, "amber")],
                       eq=f"first index with nums[i] >= 1 is {b}, so {len(nums)} - {b} = {len(nums) - b} positives"))
    steps.append(_step("two probes did the whole job", [("nums", nums, None)],
                       [("neg", a, "red"), ("pos", len(nums) - b, "amber"),
                        ("answer", max(a, len(nums) - b), "green")],
                       eq=f"max({a}, {len(nums) - b}) = {max(a, len(nums) - b)}; O(log n) instead of O(n)"))
    assert (a, len(nums) - b) == (4, 3)
    assert a == sum(1 for x in nums if x < 0) and len(nums) - b == sum(1 for x in nums if x > 0)
    return _pack("Maximum Count of Positive and Negative - two bisect probes (optimal)",
                 "nums = [-4, -3, -1, -1, 0, 2, 3, 5]", steps, max(a, len(nums) - b))


# ---------------------------------------------------------------------------
# P1b - LC 2089 Find Target Indices After Sorting Array
# ---------------------------------------------------------------------------

def _findidx_scan_brute():
    nums = [1, 2, 5, 2, 3]
    t = 2
    steps = [_step("unsorted scan for the target", [("nums", nums, None)],
               [("hits", "[]", "green")],
               eq="collect every i with nums[i] == t; correct, but the index order is the array's, not the target's")]
    hits = []
    for i, x in enumerate(nums):
        if x == t:
            hits.append(i)
        steps.append(_step(f"i = {i}, x = {x}", [("nums", nums, i)],
                           [("hits", hits, "green")],
                           eq=f"{x} == {t} -> append {i}" if x == t else f"{x} != {t}"))
    assert hits == [1, 3]
    return _pack("Find Target Indices - one scan, original positions (brute)",
                 "nums = [1, 2, 5, 2, 3], target = 2", steps, sorted(hits))


def _findidx_sort_naive():
    nums = [1, 2, 5, 2, 3]
    t = 2
    s = sorted(nums)
    steps = [_step("sort first", [("nums", nums, None), ("sorted", s, None)],
                  [("ans", "[]", "green")],
                  eq="the answer must be index-sorted after sorting the array")]
    for i, x in enumerate(s):
        if x == t:
            steps.append(_step(f"s[{i}] = {x} == t", [("sorted", s, i)],
                               [("ans", [j for j in range(i + 1) if s[j] == t], "green")],
                               eq=f"target found at sorted index {i}"))
        else:
            steps.append(_step(f"s[{i}] = {x} != t", [("sorted", s, i)],
                               [("ans", [j for j in range(len(s)) if s[j] == t and j <= i], "green")],
                               eq=""))
    ans = [i for i, x in enumerate(s) if x == t]
    assert ans == [1, 2]
    steps.append(_step("all hits collected", [("sorted", s, None)],
                      [("ans", ans, "green")],
                      eq="a full scan of the sorted array: O(n log n) sort + O(n) scan"))
    return _pack("Find Target Indices - sort, then scan every index (intermediate)",
                 "nums = [1, 2, 5, 2, 3], target = 2 -> sorted [1, 2, 2, 3, 5]", steps, str(ans))


def _findidx_bisect_opt():
    nums = [1, 2, 5, 2, 3]
    t = 2
    s = sorted(nums)
    steps = [_step("sort the array", [("sorted", s, None)],
              [("lo", "-", "indigo"), ("hi", "-", "violet")],
              eq="equal targets form one contiguous run in sorted order - hunt its ends")]
    lo = bisect.bisect_left(s, t)
    hi = bisect.bisect_right(s, t)
    ans = list(range(lo, hi))
    assert ans == [i for i, x in enumerate(s) if x == t] == [1, 2]
    steps.append(_step("bisect_left: left end", [("sorted", s, lo)],
                       [("lo", lo, "indigo"), ("hi", "-", "violet")],
                       eq=f"first index with s[i] >= {t} is {lo}"))
    steps.append(_step("bisect_right: right end", [("sorted", s, hi)],
                       [("lo", lo, "indigo"), ("hi", hi, "violet")],
                       eq=f"first index with s[i] > {t} is {hi}; the run is [{lo}, {hi})"))
    steps.append(_step("read the run off", [("sorted", s, lo)],
                       [("ans", ans, "green")],
                       eq=f"list(range({lo}, {hi})) = {ans} - two probes, no scan"))
    return _pack("Find Target Indices - bisect the run of targets (optimal)",
                 "nums = [1, 2, 5, 2, 3], target = 2 -> sorted [1, 2, 2, 3, 5]", steps, str(ans))


# ---------------------------------------------------------------------------
# P2a - LC 658 Find K Closest Elements
# ---------------------------------------------------------------------------

def _kclosest_window_brute():
    arr = [1, 2, 3, 4, 5, 6]
    k, x = 3, 3
    dist = [abs(v - x) for v in arr]
    ranked = sorted(arr, key=lambda v: (abs(v - x), v))   # closeness, ties to smaller value
    chosen = sorted(ranked[:k])
    assert chosen == [2, 3, 4]
    steps = [
        _step("rank every element by (|v - x|, v)", [("arr", arr, None)],
              [("k", k, "green")],
              eq="the k closest to x are the k elements minimising |v - x|; ties prefer the smaller v"),
        _step("read the distances", [("arr", arr, None)],
              [("|v - 3|", dist, "amber")],
              eq="distance of each value from x = 3"),
        _step("ascending by closeness", [("ranked", ranked, None)],
              [("first k", ranked[:k], "cyan")],
              eq=f"ordered by (distance, value): {ranked}; keep the first {k} = {ranked[:k]}"),
        _step("answer (re-sorted ascending)", [("chosen", chosen, None)],
              [("answer", chosen, "green")],
              eq="the k closest, returned in ascending order"),
    ]
    return _pack("Find K Closest - rank by distance (brute)",
                 "arr = [1..6], k = 3, x = 3", steps, str(chosen))


def _kclosest_shrink_mid():
    arr = [1, 2, 3, 4, 5, 6]
    k, x = 3, 3
    lo, hi = 0, len(arr) - 1
    steps = [_step("shrink the ends", [("arr", arr, lo)],
                  [("lo", lo, "indigo"), ("hi", hi, "violet")],
                  eq="while more than k elements remain, the end that is farther from x can never be in the answer")]
    while hi - lo + 1 > k:
        dl, dr = x - arr[lo], arr[hi] - x
        if dl > dr:
            lo += 1
            why = f"left end is {dl} away, right end {dr}: the left end is farther -> drop the left"
        else:
            hi -= 1
            why = f"left end is {dl} away, right end {dr}: the right end is farther or ties -> drop the right"
        steps.append(_step(f"keep [{lo}..{hi}] ({hi - lo + 1} left)", [("arr", arr, lo)],
                           [("lo", lo, "indigo"), ("hi", hi, "violet")],
                           eq=why))
    ans = arr[lo:hi + 1]
    assert ans == [2, 3, 4]
    steps.append(_step("k survivors", [("arr", arr, lo)],
                       [("answer", ans, "green")],
                       eq=f"the window [{lo}..{hi}] is the answer: {ans}"))
    return _pack("Find K Closest - shrink from the ends (intermediate)",
                 "arr = [1..6], k = 3, x = 3", steps, str(ans))


def _kclosest_bisect_opt():
    arr = [1, 2, 3, 4, 5, 6]
    k, x = 3, 3
    lo, hi = 0, len(arr) - k
    steps = [_step("search the start position", [("arr", arr, None)],
                  [("lo", lo, "indigo"), ("hi", hi, "violet")],
                  eq="the answer is arr[l..l+k-1] for some l in [0, n-k]; P(l): x - arr[l] <= arr[l+k] - x is false...true")]
    while lo < hi:
        mid = (lo + hi) // 2
        window_left = arr[mid]
        probe = arr[mid + k]
        if x - window_left > probe - x:
            lo = mid + 1
            why = f"{x} - {window_left} > {probe} - {x}: the window must start right of {mid}"
        else:
            hi = mid
            why = f"{x} - {window_left} <= {probe} - {x}: {window_left} is at least as close as {probe}, start at {mid} or left"
        steps.append(_step(f"mid = {mid}", [("arr", arr, mid)],
                           [("lo", lo, "indigo"), ("hi", hi, "violet")],
                           eq=f"compare arr[{mid}] = {window_left} with arr[{mid} + {k}] = {probe} against x = {x}: " + why))
    ans = arr[lo:lo + k]
    assert ans == [2, 3, 4]
    steps.append(_step("l found", [("arr", arr, lo)],
                       [("answer", ans, "green")],
                       eq=f"lo == hi == {lo}: {ans} - O(log n) windows tested, not O(n)"))
    return _pack("Find K Closest - binary search the left bound (optimal)",
                 "arr = [1..6], k = 3, x = 3", steps, str(ans))


# ---------------------------------------------------------------------------
# P2b - LC 1855 Maximum Distance Between a Pair of Values
# ---------------------------------------------------------------------------

def _maxdist_pair_brute():
    n1 = [55, 30, 5, 4, 2]
    n2 = [100, 20, 10, 10, 5]
    steps = [_step("two non-increasing arrays", [("nums1", n1, None), ("nums2", n2, None)],
              [("best", 0, "green")],
              eq="try every pair (i, j) with i <= j and nums1[i] <= nums2[j]")]
    best = 0
    for i in range(len(n1)):
        rowbest = max((j - i for j in range(i, len(n2)) if n1[i] <= n2[j]), default=-1)
        if rowbest > best:
            best = rowbest
        steps.append(_step(f"fix i = {i} ({n1[i]})", [("nums1", n1, i), ("nums2", n2, None)],
                           [("best j - i", rowbest, "amber"), ("best", best, "green")],
                           eq=f"all j >= {i} with {n1[i]} <= nums2[j], newest max(j-i) = {rowbest}"))
    assert best == 2
    return _pack("Maximum Pair Distance - fix i, try every j (brute)",
                 "nums1 = [55,30,5,4,2], nums2 = [100,20,10,10,5]", steps, best)


def _maxdist_two_pointers():
    n1 = [55, 30, 5, 4, 2]
    n2 = [100, 20, 10, 10, 5]
    steps = [_step("j only moves right", [("nums1", n1, None), ("nums2", n2, None)],
              [("i", 0, "indigo"), ("j", 0, "violet"), ("best", 0, "green")],
              eq="as i grows the requirement nums1[i] <= nums2[j] only gets easier, so j never needs to retreat")]
    j, best = 0, 0
    for i in range(len(n1)):
        j = max(j, i)
        while j + 1 < len(n2) and n1[i] <= n2[j + 1]:
            j += 1
        best = max(best, j - i)
        steps.append(_step(f"i = {i}: j sits at {j}", [("nums1", n1, i), ("nums2", n2, j)],
                           [("i", i, "indigo"), ("j", j, "violet"), ("best", best, "green")],
                           eq=f"{n1[i]} <= nums2[{j}] = {n2[j]}" +
                              (f", and {n1[i]} > nums2[{j + 1}]" if j + 1 < len(n2) else " (j at end)")))
    assert best == 2
    return _pack("Maximum Pair Distance - a finger that never retreats (intermediate)",
                 "nums1 = [55,30,5,4,2], nums2 = [100,20,10,10,5]", steps, best)


def _maxdist_per_i_bisect():
    n1 = [55, 30, 5, 4, 2]
    n2 = [100, 20, 10, 10, 5]
    rev = [-x for x in n2]
    steps = [_step("flip nums2 to non-decreasing", [("nums1", n1, None), ("rev", rev, None)],
              [("best", 0, "green")],
              eq="nums2 is non-increasing, so -nums2 is non-decreasing: bisect may now probe it")]
    best = 0
    for i, v in enumerate(n1):
        j = bisect.bisect_right(rev, -v) - 1
        if j >= i:
            best = max(best, j - i)
        steps.append(_step(f"i = {i} ({v})", [("nums1", n1, i), ("rev", rev, max(j, i) if j >= 0 else i)],
                           [("furthest j", j, "amber"), ("best", best, "green")],
                           eq=f"first rev[j] > {-v} is {j + 1}, so furthest valid j = {j}"))
    assert best == 2
    assert best == max(j - i for i in range(len(n1)) for j in range(i, len(n2)) if n1[i] <= n2[j])
    steps.append(_step("distance won", [("nums1", n1, None), ("nums2", n2, None)],
                       [("answer", best, "green")],
                       eq="one O(log m) probe per row: O(n log m)"))
    return _pack("Maximum Pair Distance - bisect per row (optimal)",
                 "nums1 = [55,30,5,4,2], nums2 = [100,20,10,10,5] -> rev = [-100,-20,-10,-10,-5]",
                 steps, best)


# ---------------------------------------------------------------------------
# P3a - CF 1676E Eating Queries
# ---------------------------------------------------------------------------

def _queries_greedy_brute():
    nums = [4, 3, 3, 1, 1, 4, 5, 9]
    desc = sorted(nums, reverse=True)
    x = 10
    steps = [_step("candy sugars", [("nums", nums, None)],
              [("need", x, "amber")],
              eq="query x = 10: eat the richest remaining candy until the total reaches x")]
    total, k = 0, 0
    pool = desc[:]
    while pool and total < x:
        c = pool.pop(0)
        total += c
        k += 1
        steps.append(_step(f"eat {c}", [("eaten", desc[:k], None)],
                           [("total", total, "amber"), ("candies", k, "green")],
                           eq=f"total {total - c} + {c} = {total}"))
    assert (total, k) == (14, 2)
    steps.append(_step("target reached", [("eaten", desc[:k], None)],
                       [("total", total, "amber"), ("answer", k, "green")],
                       eq=f"{k} candies give {total} >= {x}; repeat the whole walk per query -> O(n) per query"))
    return _pack("Eating Queries - greedy per query (brute)",
                 "a = [4,3,3,1,1,4,5,9], query x = 10", steps, k)


def _queries_linear_scan():
    nums = [4, 3, 3, 1, 1, 4, 5, 9]
    desc = sorted(nums, reverse=True)
    pref = list(itertools.accumulate(desc))
    steps = [_step("sort descending, build prefix sums", [("pref", pref, None)],
              [("total", pref[-1], "cyan")],
              eq="pref[k-1] = the best total with k candies; one build serves every query")]
    for x in (10, 50):
        hit = next((k for k, p in enumerate(pref, 1) if p >= x), None)
        hi_idx = (hit - 1) if hit else len(pref) - 1
        steps.append(_step(f"scan for x = {x}", [("pref", pref, hi_idx)],
                           [("answer", hit if hit else -1, "green")],
                           eq=(f"first prefix >= {x} is pref[{hit - 1}] = {pref[hit - 1]} -> k = {hit}"
                               if hit else f"even the full sum {pref[-1]} < {x} -> -1")))
    assert (next(k for k, p in enumerate(pref, 1) if p >= 10)) == 2
    return _pack("Eating Queries - one prefix array, linear scan per query (intermediate)",
                 "a = [4,3,3,1,1,4,5,9] -> pref = " + str(pref), steps, "2, -1")


def _queries_bisect_opt():
    nums = [4, 3, 3, 1, 1, 4, 5, 9]
    desc = sorted(nums, reverse=True)
    pref = list(itertools.accumulate(desc))
    steps = [_step("pref is non-decreasing", [("pref", pref, None)],
              [("lo", 0, "indigo"), ("hi", len(pref), "violet")],
              eq="P(k): pref[k] >= x is false...true - binary search the flip point")]
    for x in (10, 50):
        b = bisect.bisect_left(pref, x)
        ans = b + 1 if b < len(pref) else -1
        steps.append(_step(f"bisect_left(pref, {x})", [("pref", pref, min(b, len(pref) - 1))],
                           [("probe", b, "indigo"), ("answer", ans, "green")],
                           eq=(f"first prefix >= {x} at index {b} -> k = {b + 1}"
                               if ans != -1 else f"no prefix reaches {x} (sum {pref[-1]}) -> -1")))
    assert bisect.bisect_left(pref, 10) + 1 == 2
    assert bisect.bisect_left(pref, 50) == len(pref)
    assert [bisect.bisect_left(pref, v) + 1 if bisect.bisect_left(pref, v) < len(pref) else -1
            for v in (1, 10, 14, 15, 22, 30, 50)] == [1, 2, 2, 3, 4, 8, -1]
    return _pack("Eating Queries - binary search the prefix sums (optimal)",
                 "a = [4,3,3,1,1,4,5,9] -> pref = " + str(pref), steps, "2, -1")


# ---------------------------------------------------------------------------
# P3b - CF 1742E Scuza
# ---------------------------------------------------------------------------

def _scuza_per_query_brute():
    a = [1, 2, 1, 5]
    k = 4
    steps = [_step("climb while legs allow", [("a", a, None)],
              [("legs", k, "indigo"), ("height", 0, "green")],
              eq="query k = 4: climb step j only if a[j] <= k; stop at the first refusal")]
    height = 0
    for i, h in enumerate(a):
        if h > k:
            steps.append(_step(f"step {i + 1}: {h} > {k}", [("a", a, i)],
                               [("legs", k, "indigo"), ("height", height, "green")],
                               eq=f"blocked at step {i + 1}; total height {height}"))
            break
        height += h
        steps.append(_step(f"step {i + 1}: {h} <= {k}", [("a", a, i)],
                           [("legs", k, "indigo"), ("height", height, "green")],
                           eq=f"climb: height {height}"))
    assert height == 4
    return _pack("Scuza - walk the stairs per query (brute)",
                 "a = [1, 2, 1, 5], query k = 4", steps, height)


def _scuza_precompute_linear():
    a = [1, 2, 1, 5]
    pref = list(itertools.accumulate(a))
    pm = [max(a[:i + 1]) for i in range(len(a))]
    steps = [_step("build once: prefix sums and prefix maxima",
                   [("pref", pref, None), ("pm", pm, None)],
              [("readable", "yes", "cyan")],
              eq="pm[i] = tallest step among the first i+1; a prefix is climbable iff pm <= k")]
    for k in (1, 9):
        i = 0
        while i < len(pm) and pm[i] <= k:
            i += 1
        ans = pref[i - 1] if i else 0
        steps.append(_step(f"k = {k}: walk pm", [("pm", pm, max(i - 1, 0))],
                           [("steps", i, "amber"), ("answer", ans, "green")],
                           eq=f"pm[{i - 1}] = {pm[i - 1]} <= {k} < pm[{i}]" +
                              (f" (={pm[i]})" if i < len(pm) else " (end)") +
                              f" -> height {ans}" if i else f"even pm[0] = {pm[0]} > {k} -> 0"))
    assert pref[2] == 4 and pref[3] == 9
    return _pack("Scuza - precompute, then scan per query (intermediate)",
                 "a = [1,2,1,5] -> pref = [1,3,4,9], pm = [1,2,2,5]", steps, "1, 9")


def _scuza_bisect_opt():
    a = [1, 2, 1, 5]
    pref = list(itertools.accumulate(a))
    pm = [max(a[:i + 1]) for i in range(len(a))]
    steps = [_step("pm is non-decreasing", [("pm", pm, None)],
              [("lo", 0, "indigo"), ("hi", len(pm), "violet")],
              eq="P(i): pm[i] <= k is true...false - bisect the last true, answer is pref there")]
    for k in (1, 2, 9):
        i = bisect.bisect_right(pm, k)
        ans = pref[i - 1] if i else 0
        steps.append(_step(f"k = {k}: bisect_right(pm, {k})", [("pm", pm, i - 1 if i else 0)],
                           [("first > k", i, "indigo"), ("answer", ans, "green")],
                           eq=f"steps 1..{i} climbable -> height pref[{i - 1}] = {ans}" if i
                              else f"no step has pm <= {k} -> 0"))
    assert [pref[bisect.bisect_right(pm, k) - 1] if bisect.bisect_right(pm, k) else 0
            for k in (1, 2, 4, 9, 10)] == [1, 4, 4, 9, 9]
    brute = [pref[i - 1] if i else 0 for i in
             (next((t for t in range(len(a)) if max(a[:t + 1]) > k), len(a)) for k in (1, 2, 9))]
    assert [pref[bisect.bisect_right(pm, k) - 1] if bisect.bisect_right(pm, k) else 0
            for k in (1, 2, 9)] == brute == [1, 4, 9]
    return _pack("Scuza - binary search the prefix maxima (optimal)",
                 "a = [1,2,1,5] -> pref = [1,3,4,9], pm = [1,2,2,5]; queries 1,2,9", steps, "1, 4, 9")


# ---------------------------------------------------------------------------
# P4a - LC 1608 Special Array With X Elements Greater Than or Equal X
# ---------------------------------------------------------------------------

def _special_count(nums, x):
    return sum(1 for v in nums if v >= x)


def _special_scan_brute():
    nums = [0, 4, 3, 0, 4]
    steps = [_step("candidates 0..n", [("nums", nums, None)],
              [("x", "-", "indigo"), ("answer", -1, "green")],
              eq="x is a count of elements, so 0 <= x <= n; count nums[i] >= x for each x")]
    ans = -1
    for x in range(len(nums) + 1):
        c = _special_count(nums, x)
        if c == x and ans == -1:
            ans = x
        steps.append(_step(f"x = {x}", [("nums", nums, None)],
                           [("c(x)", c, "amber"), ("answer", ans, "green")],
                           eq=f"{c} elements are >= {x}" + ("  -> special" if c == x else f", {c} != {x}")))
    assert ans == 3
    return _pack("Special Array - count for every x (brute)",
                 "nums = [0, 4, 3, 0, 4]", steps, ans)


def _special_probe_each():
    nums = [0, 4, 3, 0, 4]
    s = sorted(nums)
    n = len(s)
    steps = [_step("sort once", [("s", s, None)],
              [("n", n, "indigo")],
              eq="in the sorted copy the elements >= x are a suffix: c(x) = n - bisect_left(s, x)")]
    ans = -1
    for x in range(n + 1):
        b = bisect.bisect_left(s, x)
        c = n - b
        if c == x and ans == -1:
            ans = x
        steps.append(_step(f"x = {x}: probe", [("s", s, min(b, n - 1))],
                           [("c(x)", c, "amber"), ("answer", ans, "green")],
                           eq=f"{n} - bisect_left(s, {x}) = {n} - {b} = {c}" + (" = x" if c == x else f" != {x}")))
    assert ans == 3
    return _pack("Special Array - sort once, one probe per x (intermediate)",
                 "nums = [0,4,3,0,4] -> s = [0,0,3,4,4]", steps, ans)


def _special_search_x():
    nums = [0, 4, 3, 0, 4]
    s = sorted(nums)
    n = len(s)

    def c(x):
        return n - bisect.bisect_left(s, x)

    lo, hi = 0, n
    steps = [_step("search x in [0, n]", [("s", s, None)],
              [("lo", lo, "indigo"), ("hi", hi, "violet")],
              eq="c(x) - x strictly falls, so Q(x): c(x) <= x is false...true; find its first true")]
    while lo < hi:
        mid = (lo + hi) // 2
        cm = c(mid)
        if cm <= mid:
            hi = mid
            why = f"c({mid}) = {cm} <= {mid}: first true is at {mid} or left"
        else:
            lo = mid + 1
            why = f"c({mid}) = {cm} > {mid}: first true is right of {mid}"
        steps.append(_step(f"mid = {mid}", [("s", s, min(bisect.bisect_left(s, mid), n - 1))],
                           [("lo", lo, "indigo"), ("hi", hi, "violet")], eq=why))
    ans = lo if c(lo) == lo else -1
    steps.append(_step("check the flip", [("s", s, None)],
                       [("x", lo, "indigo"), ("c(x)", c(lo), "amber"), ("answer", ans, "green")],
                       eq=f"c({lo}) = {c(lo)}" + (" = x: special" if ans != -1 else " != x: not special")))
    assert ans == 3 == next((x for x in range(n + 1) if _special_count(nums, x) == x), -1)
    return _pack("Special Array - binary search on x (optimal)",
                 "nums = [0,4,3,0,4] -> s = [0,0,3,4,4]", steps, ans)


# ---------------------------------------------------------------------------
# P4b - LC 1300 Sum of Mutated Array Closest to Target
# ---------------------------------------------------------------------------

def _capmin_brute():
    arr = [4, 9, 3]
    target = 10
    steps = [_step("try every cap", [("arr", arr, None)],
              [("best cap", "-", "green")],
              eq="S(v) = sum(min(a, v)) grows with v; brute force walks v = 0, 1, 2, ...")]
    best, bv = math.inf, -1
    for v in range(0, max(arr) + 1):
        s = sum(min(a, v) for a in arr)
        d = abs(s - target)
        if d < best:
            best, bv = d, v
        if v <= 4:                           # keep the board inside 8 frames
            steps.append(_step(f"cap v = {v}", [("arr", arr, None)],
                               [("sum", s, "amber"), ("diff", d, "cyan"), ("best", bv, "green")],
                               eq=f"S({v}) = {s}, |{s} - 10| = {d}"))
    assert bv == 3
    return _pack("Mutated Array Closest - walk the caps (brute)",
                 "arr = [4,9,3], target = 10", steps, bv)


def _capmin_sorted_linear():
    arr = [4, 9, 3]
    target = 10

    def S(v):
        return sum(min(a, v) for a in arr)

    lo, hi = 0, max(arr)
    steps = [_step("binary search the cap value", [("arr", arr, None)],
              [("lo", lo, "indigo"), ("hi", hi, "violet")],
              eq="S(v) = sum(min(a, v)) is non-decreasing in v; P(v): S(v) >= target flips once")]
    while lo < hi:
        mid = (lo + hi) // 2
        total = S(mid)
        if total >= target:
            hi = mid
            why = f"S({mid}) = {total} >= {target}: first reach is at {mid} or left"
        else:
            lo = mid + 1
            why = f"S({mid}) = {total} < {target}: first reach is right of {mid}"
        steps.append(_step(f"mid = {mid}", [("arr", arr, None)],
                           [("S(mid)", total, "amber"), ("lo", lo, "indigo"), ("hi", hi, "violet")],
                           eq=f"scan: " + why))
    c1, c2 = hi - 1, hi
    d1 = abs(S(c1) - target) if c1 >= 0 else math.inf
    d2 = abs(S(c2) - target)
    best = c1 if d1 <= d2 else c2
    steps.append(_step("the seam caps", [("arr", arr, None)],
                       [(f"cap {c1}", f"S = {S(c1)}, diff {d1}", "amber"),
                        (f"cap {c2}", f"S = {S(c2)}, diff {d2}", "cyan"),
                        ("answer", best, "green")],
                       eq="S crossed the target between hi-1 and hi; compare them, smaller wins ties"))
    ref = min(range(0, max(arr) + 1), key=lambda v: (abs(S(v) - target), v))
    assert best == ref == 3, (best, ref)
    return _pack("Mutated Array Closest - bisect the cap, scan each sum (intermediate)",
                 "arr = [4,9,3], target = 10", steps, best)


def _capmin_search_opt():
    arr = [1, 2, 4, 9, 3]
    target = 15
    s = sorted(arr)
    pref = list(itertools.accumulate(s))
    n = len(s)

    def S(v):
        b = bisect.bisect_right(s, v)
        if b == 0:
            return n * v
        return pref[b - 1] + (n - b) * v

    steps = [_step("S(v) is non-decreasing", [("s", s, None), ("pref", pref, None)],
              [("lo", 0, "indigo"), ("hi", s[-1], "violet")],
              eq="P(v): S(v) >= target is false...true over caps 0..max; binary search the flip point")]
    lo, hi = 0, s[-1]
    while lo < hi:
        mid = (lo + hi) // 2
        total = S(mid)
        if total >= target:
            hi = mid
        else:
            lo = mid + 1
        steps.append(_step(f"mid = {mid}", [("s", s, bisect.bisect_right(s, mid) - 1)],
                           [("S(mid)", total, "amber"), ("lo", lo, "indigo"), ("hi", hi, "violet")],
                           eq=f"S({mid}) = {total}" +
                              (" >= 15 -> caps right of here cost more" if total >= target
                               else " < 15 -> caps up to here are too small")))
    c1, c2 = hi - 1, hi                      # the two seam caps: last short, first reaches
    d1, d2 = abs(S(c1) - target), abs(S(c2) - target)
    best = c1 if d1 <= d2 else c2            # '<=': exact ties go to the smaller cap
    steps.append(_step("the two seam caps", [("s", s, bisect.bisect_right(s, c2) - 1)],
                       [(f"cap {c1}", f"S = {S(c1)}, diff {d1}", "amber"),
                        (f"cap {c2}", f"S = {S(c2)}, diff {d2}", "cyan"),
                        ("answer", best, "green")],
                       eq="S jumps over the target between hi-1 and hi: compare the seam, "
                          "smaller cap wins exact ties"))
    ref = min(range(0, max(arr) + 1), key=lambda v: (abs(sum(min(a, v) for a in arr) - target), v))
    assert best == ref == 5, (best, ref)
    return _pack("Mutated Array Closest - search on the cap value (optimal)",
                 "arr = [1,2,4,9,3] -> s = [1,2,3,4,9], target = 15", steps, best)


def _findidx_count_opt():
    nums = [1, 2, 5, 2, 3]
    t = 2
    steps = [_step("count below / equal", [("nums", nums, None)],
              [("below", 0, "indigo"), ("equal", 0, "amber")],
              eq="no sort needed: the run starts after everything below t and lasts as long as t occurs")]
    below = equal = 0
    for i, x in enumerate(nums):
        if x < t:
            below += 1
        elif x == t:
            equal += 1
        steps.append(_step(f"x = {x}", [("nums", nums, i)],
                           [("below", below, "indigo"), ("equal", equal, "amber")],
                           eq=(f"{x} < {t} -> below := {below}" if x < t else
                               f"{x} == {t} -> equal := {equal}" if x == t else
                               f"{x} > {t} -> ignored")))
    assert (below, equal) == (1, 2)
    steps.append(_step("read the run off", [("nums", nums, None)],
                       [("answer", list(range(below, below + equal)), "green")],
                       eq=f"range({below}, {below} + {equal}) = {list(range(below, below + equal))}"))
    return _pack("Find Target Indices - count around the target, no sort (optimal)",
                 "nums = [1, 2, 5, 2, 3], target = 2", steps,
                 str(list(range(below, below + equal))))


_ALL = None


def get(problem, form):
    """Return ``(title, input line, steps, final)`` for one solution form."""
    global _ALL
    if _ALL is None:
        _ALL = {
            # P1a - LC 2529: linear counts -> count the runs -> two bisect probes
            ("signs", "brute"): _signs_scan_brute(),
            ("signs", "naive"): _signs_count_naive(),
            ("signs", "bisect"): _signs_bisect_opt(),
            # P1b - LC 2089: scan -> sort + scan -> bisect the run
            ("findidx", "brute"): _findidx_scan_brute(),
            ("findidx", "naive"): _findidx_sort_naive(),
            ("findidx", "bisect"): _findidx_bisect_opt(),
            ("findidx", "count"): _findidx_count_opt(),
            # P2a - LC 658: window brute -> shrink ends -> bisect the left bound
            ("kclosest", "brute"): _kclosest_window_brute(),
            ("kclosest", "shrink"): _kclosest_shrink_mid(),
            ("kclosest", "bisect"): _kclosest_bisect_opt(),
            # P2b - LC 1855: pair brute -> two pointers -> per-i bisect
            ("maxdist", "brute"): _maxdist_pair_brute(),
            ("maxdist", "tp"): _maxdist_two_pointers(),
            ("maxdist", "bisect"): _maxdist_per_i_bisect(),
            # P3a - CF 1676E: greedy per query -> prefix + scan -> prefix + bisect
            ("queries", "greedy"): _queries_greedy_brute(),
            ("queries", "scan"): _queries_linear_scan(),
            ("queries", "bisect"): _queries_bisect_opt(),
            # P3b - CF 1742E: walk per query -> precompute + scan -> bisect prefix maxima
            ("scuza", "brute"): _scuza_per_query_brute(),
            ("scuza", "scan"): _scuza_precompute_linear(),
            ("scuza", "bisect"): _scuza_bisect_opt(),
            # P4a - LC 1608: count for every x -> one probe per x -> binary search on x
            ("special", "brute"): _special_scan_brute(),
            ("special", "probe"): _special_probe_each(),
            ("special", "bisect"): _special_search_x(),
            # P4b - LC 1300: walk the caps -> sorted linear -> search on the cap
            ("capmin", "brute"): _capmin_brute(),
            ("capmin", "linear"): _capmin_sorted_linear(),
            ("capmin", "search"): _capmin_search_opt(),
        }
    return _ALL[(problem, form)]
