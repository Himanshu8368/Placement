"""Step frames for the two-pointers chapter.

Drawing lives in ``dsa_viz.story``. ``get`` returns data only.
Each story walks the chapter's own sample, at most eight frames, and the
numbers come from running the algorithm - never from a hand-typed table.
A two-pointer frame shows the array row plus the rolling pointers as chips,
so the reader watches ``lo`` and ``hi`` converge in step with the arithmetic.
"""
from __future__ import annotations

import bisect
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
    """One frame: an array row (or two) plus the moving pointers as chips."""
    frame = row_step(title, rows, eq=eq)
    if chips:
        frame["chips"] = chip_step(title, chips, eq="")["chips"]
    return frame


def _two_sum2():
    a, target = [2, 7, 11, 15], 9
    lo, hi, steps = 0, len(a) - 1, []
    final = None
    while lo < hi:
        s = a[lo] + a[hi]
        if s == target:
            steps.append(_step(f"lo = {lo}, hi = {hi}", [("a", a, None)],
                               [("lo", lo, "indigo"), ("hi", hi, "violet"), ("sum", s, "green")],
                               eq=f"{a[lo]} + {a[hi]} = {s} = target -> return [{lo + 1}, {hi + 1}]"))
            final = [lo + 1, hi + 1]
            break
        move = "raise lo" if s < target else "lower hi"
        steps.append(_step(f"lo = {lo}, hi = {hi}", [("a", a, None)],
                           [("lo", lo, "indigo"), ("hi", hi, "violet"), ("sum", s, "amber")],
                           eq=f"{a[lo]} + {a[hi]} = {s} {'<' if s < target else '>'} {target} -> {move}"))
        if s < target:
            lo += 1
        else:
            hi -= 1
    return _pack("Two Sum II - sorted, discard the end that cannot help",
                 "a = [2, 7, 11, 15], target = 9", steps, final)


def _container_brute():
    h = [1, 8, 6, 2, 5]
    n, best, steps = len(h), 0, []
    for i in range(n):
        for j in range(i + 1, n):
            area = (j - i) * min(h[i], h[j])
            if area > best:
                prev = best
                best = area
                if len(steps) < 8:
                    steps.append(_step(f"pair i = {i}, j = {j}", [("h", h, None)],
                                       [("i", i, "indigo"), ("j", j, "violet"),
                                        ("area", area, "amber"), ("best", best, "green")],
                                       eq=f"width {j - i} x min({h[i]},{h[j]}) = {area} > best {prev} -> new best"))
    return _pack("Container With Most Water - every pair, keeping the best improvement",
                 "h = [1, 8, 6, 2, 5]", steps, best)


def _container():
    h = [1, 8, 6, 2, 5]
    lo, hi, best, steps = 0, len(h) - 1, 0, []
    while lo < hi:
        area = (hi - lo) * min(h[lo], h[hi])
        best = max(best, area)
        move = "raise lo" if h[lo] < h[hi] else "lower hi"
        steps.append(_step(f"lo = {lo}, hi = {hi}", [("h", h, None)],
                           [("lo", lo, "indigo"), ("hi", hi, "violet"), ("area", area, "amber"),
                            ("best", best, "green")],
                           eq=f"width {hi - lo} x min({h[lo]},{h[hi]}) = {area}; move the shorter wall ({move})"))
        if h[lo] < h[hi]:
            lo += 1
        else:
            hi -= 1
    return _pack("Container With Most Water - the shorter wall is the one to move",
                 "h = [1, 8, 6, 2, 5]", steps, best)


def _merge_front():
    nums1, m, nums2, n = [1, 2, 3], 3, [2, 5, 6], 3
    out, i, j, steps = [], 0, 0, []
    while i < m or j < n:
        if j >= n or (i < m and nums1[i] <= nums2[j]):
            val, src, i = nums1[i], f"nums1[{i}]", i + 1
        else:
            val, src, j = nums2[j], f"nums2[{j}]", j + 1
        out.append(val)
        steps.append(_step(f"write out[{len(out) - 1}]", [("nums1", nums1, i), ("nums2", nums2, j),
                                                          ("out", list(out), None)],
                           [("i", i, "indigo"), ("j", j, "violet"), ("head", val, "green")],
                           eq=f"smallest unread is {src} = {val} -> append to the fresh array"))
    return _pack("Merge Sorted Array - classic front merge into a fresh array (O(m+n) space)",
                 "nums1 = [1,2,3], nums2 = [2,5,6]", steps, out)


def _merge_back():
    nums1, m, nums2, n = [1, 2, 3, 0, 0, 0], 3, [2, 5, 6], 3
    i, j, p, steps = m - 1, n - 1, m + n - 1, []
    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            val, i = nums1[i], i - 1
            src = f"nums1[{i + 1}] = {val}"
        else:
            val, j = nums2[j], j - 1
            src = f"nums2[{j + 1}] = {val}"
        nums1[p] = val
        steps.append(_step(f"write at p = {p}", [("nums1", list(nums1), p), ("nums2", nums2, None)],
                           [("i", i, "indigo"), ("j", j, "violet"), ("p", p, "green")],
                           eq=f"largest free is {val} (from {src}) -> place at the back"))
        p -= 1
    return _pack("Merge Sorted Array - fill from the back so nothing is overwritten",
                 "nums1 = [1,2,3,_,_,_], nums2 = [2,5,6]", steps, list(nums1))


def _squares_back():
    nums = [-4, -1, 0, 3, 10]
    n = len(nums)
    out, lo, hi, p, steps = [None] * n, 0, n - 1, n - 1, []
    while lo <= hi:
        a, b = nums[lo] * nums[lo], nums[hi] * nums[hi]
        if a >= b:
            out[p], lo = a, lo + 1
            pick = f"{nums[lo - 1]}^2 = {a} (left)"
        else:
            out[p], hi = b, hi - 1
            pick = f"{nums[hi + 1]}^2 = {b} (right)"
        steps.append(_step(f"fill out[{p}]", [("nums", nums, None), ("out", list(out), p)],
                           [("lo", lo - (1 if a >= b else 0), "indigo"), ("hi", hi + (0 if a >= b else 1), "violet"),
                            ("p", p, "green")],
                           eq=f"bigger square goes to the back: {pick}"))
        p -= 1
    return _pack("Squares of a Sorted Array - the squares are sorted from the outside in",
                 "nums = [-4, -1, 0, 3, 10]", steps, out)


def _boats():
    people, limit = sorted([3, 2, 2, 1]), 3
    lo, hi, boats, steps = 0, len(people) - 1, 0, []
    while lo <= hi:
        if people[lo] + people[hi] <= limit:
            note = f"{people[lo]} + {people[hi]} <= {limit}: share a boat"
            lo += 1
        else:
            note = f"lightest + heaviest > {limit}: heaviest rides alone"
        hi -= 1
        boats += 1
        steps.append(_step(f"boat #{boats}", [("people", people, hi)],
                           [("lo", lo, "indigo"), ("hi", hi, "violet"), ("boats", boats, "green")],
                           eq=note))
    return _pack("Boats to Save People - pair the lightest with the heaviest",
                 "people = [3, 2, 2, 1], limit = 3", steps, boats)


def _pairs_bisect():
    d = sorted(x - y for x, y in zip([7, 8, 4, 1], [1, 5, 3, 6]))
    n, c, steps = len(d), 0, []
    steps.append(_step("transform d[i] = a[i] - b[i], then sort", [("d", d, None)],
                       [("note", "per-left bisect", "muted")],
                       eq="for each lo count partners via bisect_left(d, 1 - d[lo], lo + 1)"))
    for lo in range(n - 1):
        k = bisect.bisect_left(d, 1 - d[lo], lo + 1)
        add = n - k if k > lo else 0
        c += add
        steps.append(_step(f"left lo = {lo} (d = {d[lo]})", [("d", d, lo)],
                           [("lo", lo, "indigo"), ("need", 1 - d[lo], "violet"),
                            ("k", k, "amber"), ("pairs", c, "green")],
                           eq=f"first j with {d[lo]} + d[j] > 0 is {k}; partners = {n - k} -> +{add}"))
    return _pack("Pair of Topics - count positive pairs by a bisect per left (O(n log n))",
                 "a = [7,8,4,1], b = [1,5,3,6]  (d = [-5,1,3,6] sorted)", steps, c)


def _pair_topics():
    a, b = [7, 8, 4, 1], [1, 5, 3, 6]
    d = sorted(x - y for x, y in zip(a, b))
    lo, hi, c, steps = 0, len(d) - 1, 0, []
    steps.append(_step("transform d[i] = a[i] - b[i], then sort", [("d", d, None)],
                       [("lo", lo, "indigo"), ("hi", hi, "violet"), ("pairs", c, "muted")],
                       eq="count pairs with d[lo] + d[hi] > 0"))
    while lo < hi:
        s = d[lo] + d[hi]
        if s > 0:
            c += hi - lo
            eq = f"{d[lo]} + {d[hi]} = {s} > 0: this hi pairs with every lo..{hi - 1} -> +{hi - lo}"
            hi -= 1
        else:
            eq = f"{d[lo]} + {d[hi]} = {s} <= 0: this lo too small -> raise lo"
            lo += 1
        steps.append(_step(f"lo = {lo - (1 if s <= 0 else 0)}, hi = {hi + (1 if s <= 0 else 0)}",
                           [("d", d, None)],
                           [("lo", lo, "indigo"), ("hi", hi, "violet"), ("pairs", c, "green")], eq=eq))
    return _pack("Pair of Topics - count positive pairs with opposite ends",
                 "a = [7,8,4,1], b = [1,5,3,6]  (d = [6,3,1,-5])", steps, c)


def _summary_runs():
    nums = [0, 1, 2, 4, 5, 7]
    res, i, n, steps = [], 0, len(nums), []
    while i < n:
        j = i
        while j + 1 < n and nums[j + 1] == nums[j] + 1:
            j += 1
        res.append(f"{nums[i]}" if i == j else f"{nums[i]}->{nums[j]}")
        steps.append(_step(f"run from index {i}", [("nums", nums, j)],
                           [("i", i, "indigo"), ("j", j, "violet")],
                           eq=f"{nums[i]}..{nums[j]} consecutive -> '{res[-1]}'"))
        i = j + 1
    return _pack("Summary Ranges - the group pointer jumps a whole consecutive block",
                 "nums = [0, 1, 2, 4, 5, 7]", steps, res)


def _alt_runs():
    a = [1, 2, 3, -1, -2, 3]
    total, i, n, steps = 0, 0, len(a), []
    while i < n:
        j = i
        while j + 1 < n and (a[j + 1] > 0) == (a[i] > 0):
            j += 1
        m = max(a[i:j + 1])
        total += m
        sign = "positive" if a[i] > 0 else "negative"
        steps.append(_step(f"same-sign run at {i}", [("a", a, j)],
                           [("i", i, "indigo"), ("j", j, "violet"), ("pick", m, "green"),
                            ("total", total, "amber")],
                           eq=f"{sign} run {a[i:j + 1]} -> keep the max {m}"))
        i = j + 1
    return _pack("Alternating Subsequence - one best element per same-sign run",
                 "a = [1, 2, 3, -1, -2, 3]", steps, total)


def _two2_bs():
    a, target = [2, 7, 11, 15], 9
    steps, final = [], None
    for i in range(len(a) - 1):
        need = target - a[i]
        j = bisect.bisect_left(a, need, i + 1)
        hit = j < len(a) and a[j] == need
        steps.append(_step(f"fix lo = {i}, bisect for {need}", [("a", a, i)],
                           [("lo", i, "indigo"), ("need", need, "violet"),
                            ("index", j, "amber"),
                            ("hit", "yes" if hit else "no", "green" if hit else "muted")],
                           eq=(f"binary search {need} in a[{i + 1}:]: found at {j} -> return [{i + 1}, {j + 1}]"
                               if hit else f"no {need} in a[{i + 1}:]: advance lo")))
        if hit:
            final = [i + 1, j + 1]
            break
    return _pack("Two Sum II - fix the left end, binary-search the partner (O(n log n))",
                 "a = [2, 7, 11, 15], target = 9", steps, final)


def _squares_sort():
    nums = [-4, -1, 0, 3, 10]
    sq = [x * x for x in nums]
    steps = [_step("square every element", [("nums", nums, None), ("sq", sq, None)],
                   [("note", "order lost", "muted")],
                   eq="sq[i] = nums[i]^2 - the negatives at both ends scatter the order"),
             _step("sort the squares", [("nums", nums, None), ("sq", sorted(sq), None)],
                   [("note", "O(n log n)", "amber")],
                   eq="sorting the squared array is correct but costs an extra log factor")]
    return _pack("Squares of a Sorted Array - square then sort (the O(n log n) route)",
                 "nums = [-4, -1, 0, 3, 10]", steps, sorted(sq))


def _three_bs():
    nums = sorted([-1, 0, 1, 2, -1, -4])
    n, seen, steps = len(nums), set(), []
    for i in range(n - 2):
        if i and nums[i] == nums[i - 1]:
            continue
        for j in range(i + 1, n - 1):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            need = -nums[i] - nums[j]
            k = bisect.bisect_left(nums, need, j + 1)
            hit = k < n and nums[k] == need
            if hit:
                seen.add((nums[i], nums[j], nums[k]))
            steps.append(_step(f"fix pair ({nums[i]}, {nums[j]})", [("nums", nums, None)],
                               [("i", i, "indigo"), ("j", j, "violet"), ("need", need, "amber"),
                                ("found", len(seen), "green")],
                               eq=(f"bisect {need} in a[{j + 1}:]: at {k} -> triple [{nums[i]}, {nums[j]}, {need}]"
                                   if hit else f"no {need} in a[{j + 1}:]: skip this pair")))
    return _pack("3Sum - fix two, binary-search the third (O(n^2 log n))",
                 "nums = [-1, 0, 1, 2, -1, -4] (sorted [-4, -1, -1, 0, 1, 2])", steps,
                 [list(t) for t in sorted(seen)])


def _three_squeeze():
    nums = sorted([-1, 0, 1, 2, -1, -4])
    n, res, steps = len(nums), [], []
    for i in range(n - 2):
        if nums[i] > 0:
            break
        if i and nums[i] == nums[i - 1]:
            continue
        lo, hi = i + 1, n - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s < 0:
                eq = f"{nums[i]} + {nums[lo]} + {nums[hi]} = {s} < 0 -> raise lo"
                lo += 1
            elif s > 0:
                eq = f"{nums[i]} + {nums[lo]} + {nums[hi]} = {s} > 0 -> lower hi"
                hi -= 1
            else:
                res.append([nums[i], nums[lo], nums[hi]])
                eq = f"{nums[i]} + {nums[lo]} + {nums[hi]} = 0 -> emit, skip equal ends"
                lo += 1
                hi -= 1
                while lo < hi and nums[lo] == nums[lo - 1]:
                    lo += 1
                while lo < hi and nums[hi] == nums[hi + 1]:
                    hi -= 1
            steps.append(_step(f"pivot i = {i} ({nums[i]})", [("nums", nums, None)],
                               [("i", i, "indigo"), ("lo", lo, "violet"), ("hi", hi, "muted"),
                                ("found", len(res), "green")], eq=eq))
    return _pack("3Sum - fix one, squeeze the tail (O(n^2))",
                 "nums = [-1, 0, 1, 2, -1, -4] (sorted [-4, -1, -1, 0, 1, 2])", steps, res)


def _close_bs():
    nums = sorted([-1, 2, 1, -4])
    target, n = 1, len(nums)
    best = nums[0] + nums[1] + nums[2]
    bd = abs(best - target)
    steps = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            need = target - nums[i] - nums[j]
            k = bisect.bisect_left(nums, need, j + 1)
            for cand in (k - 1, k):
                if j < cand < n:
                    s = nums[i] + nums[j] + nums[cand]
                    if abs(s - target) < bd:
                        bd, best = abs(s - target), s
            steps.append(_step(f"fix pair ({nums[i]}, {nums[j]})", [("nums", nums, None)],
                               [("i", i, "indigo"), ("j", j, "violet"), ("need", need, "amber"),
                                ("best", best, "green"), ("dist", bd, "muted")],
                               eq=f"bisect {need} in a[{j + 1}:]; probe neighbours -> best sum {best} (|dist| {bd})"))
    return _pack("3Sum Closest - fix two, bisect and probe neighbours (O(n^2 log n))",
                 "nums = [-1, 2, 1, -4], target = 1 (sorted [-4, -1, 1, 2])", steps, best)


def _close_squeeze():
    nums = sorted([-1, 2, 1, -4])
    target, n = 1, len(nums)
    best = nums[0] + nums[1] + nums[2]
    steps, done = [], False
    for i in range(n - 2):
        lo, hi = i + 1, n - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if abs(s - target) < abs(best - target):
                best = s
            if s == target:
                eq = f"{nums[i]} + {nums[lo]} + {nums[hi]} = {s} = target -> exact hit"
                steps.append(_step(f"pivot i = {i}", [("nums", nums, None)],
                                   [("lo", lo, "indigo"), ("hi", hi, "violet"), ("best", best, "green")], eq=eq))
                done = True
                break
            eq = (f"{s} vs target {target}: |{s} - {target}| = {abs(s - target)}, {'raise lo' if s < target else 'lower hi'}")
            if s < target:
                lo += 1
            else:
                hi -= 1
            steps.append(_step(f"pivot i = {i} ({nums[i]})", [("nums", nums, None)],
                               [("lo", lo, "indigo"), ("hi", hi, "violet"), ("best", best, "green"),
                                ("dist", abs(best - target), "muted")], eq=eq))
        if done:
            break
    return _pack("3Sum Closest - fix one, squeeze by distance (O(n^2))",
                 "nums = [-1, 2, 1, -4], target = 1 (sorted [-4, -1, 1, 2])", steps, best)


def _pairs_bs():
    a = sorted([3, 5, 2, 1, 2, 4])
    x, n = 6, len(a)
    c, steps = 0, []
    for lo in range(n - 1):
        hi = bisect.bisect_right(a, x - a[lo]) - 1
        add = hi - lo if hi > lo else 0
        c += add
        steps.append(_step(f"left lo = {lo} (a = {a[lo]})", [("a", a, lo)],
                           [("lo", lo, "indigo"), ("limit", x - a[lo], "violet"),
                            ("hi", hi, "amber"), ("count", c, "green")],
                           eq=f"largest index with a[hi] <= {x - a[lo]} is {hi}; partners = {add}"))
    return _pack("Number of Pairs - count <= bound by a binary search per left (O(n log n))",
                 "a = [3, 5, 2, 1, 2, 4] (sorted [1, 2, 2, 3, 4, 5]), bound = 6", steps, c)


def _pairs_count():
    a = sorted([3, 5, 2, 1, 2, 4])
    x, n = 6, len(a)
    lo, hi, c, steps = 0, n - 1, 0, []
    while lo < hi:
        if a[lo] + a[hi] <= x:
            c += hi - lo
            eq = f"{a[lo]} + {a[hi]} = {a[lo] + a[hi]} <= {x}: lo pairs with all {hi - lo} between -> +{hi - lo}"
            lo += 1
        else:
            eq = f"{a[lo]} + {a[hi]} = {a[lo] + a[hi]} > {x}: lower hi"
            hi -= 1
        steps.append(_step(f"lo = {lo}, hi = {hi}", [("a", a, None)],
                           [("lo", lo, "indigo"), ("hi", hi, "violet"), ("count", c, "green")], eq=eq))
    return _pack("Number of Pairs - count <= 6 by batching in one squeeze (count_le engine)",
                 "a = [3, 5, 2, 1, 2, 4] (sorted), bound = 6", steps, c)


def _fair_bs():
    a = sorted([0, 1, 7, 4, 4, 5])
    x, n = 6, len(a)
    c, steps = 0, []
    for lo in range(n - 1):
        hi = bisect.bisect_right(a, x - a[lo]) - 1
        add = hi - lo if hi > lo else 0
        c += add
        steps.append(_step(f"left lo = {lo} (a = {a[lo]})", [("a", a, lo)],
                           [("lo", lo, "indigo"), ("limit", x - a[lo], "violet"),
                            ("hi", hi, "amber"), ("count", c, "green")],
                           eq=f"largest index with a[hi] <= {x - a[lo]} is {hi}; partners = {add}"))
    return _pack("Fair Pairs - count <= upper by a binary search per left (count_le(6) = 7)",
                 "nums = [0, 1, 7, 4, 4, 5] (sorted), bound = 6", steps, c)


def _fair_count():
    a = sorted([0, 1, 7, 4, 4, 5])
    x, n = 6, len(a)
    lo, hi, c, steps = 0, n - 1, 0, []
    while lo < hi:
        if a[lo] + a[hi] <= x:
            c += hi - lo
            eq = f"{a[lo]} + {a[hi]} = {a[lo] + a[hi]} <= {x}: bank {hi - lo} partners"
            lo += 1
        else:
            eq = f"{a[lo]} + {a[hi]} = {a[lo] + a[hi]} > {x}: lower hi"
            hi -= 1
        steps.append(_step(f"lo = {lo}, hi = {hi}", [("a", a, None)],
                           [("lo", lo, "indigo"), ("hi", hi, "violet"), ("count", c, "green")], eq=eq))
    return _pack("Fair Pairs - count <= 6 by squeeze (count_le(6) = 7; band = 7 - count_le(2))",
                 "nums = [0, 1, 7, 4, 4, 5] (sorted), bound = 6", steps, c)


def _minpair():
    nums = [3, 5, 4, 2, 4, 6]
    a = sorted(nums)
    n, best, steps = len(a), 0, []
    for i in range(n // 2):
        s_ = a[i] + a[n - 1 - i]
        best = max(best, s_)
        steps.append(_step(f"pair {i}: lo = {i}, hi = {n - 1 - i}", [("a", a, None)],
                           [("lo", i, "indigo"), ("hi", n - 1 - i, "violet"),
                            ("sum", s_, "amber"), ("max", best, "green")],
                           eq=f"smallest left {a[i]} with largest left {a[n - 1 - i]}: {s_}; max so far {best}"))
    return _pack("Minimize Maximum Pair Sum - pair the smallest with the largest",
                 "nums = [3, 5, 4, 2, 4, 6] (sorted [2, 3, 4, 4, 5, 6])", steps, best)


def _fair_sweep():
    a = sorted([0, 1, 7, 4, 4, 5])
    lower, upper, n = 3, 6, 6
    L, R, c, steps = n, n - 1, 0, []
    for i in range(n):
        while L > 0 and a[L - 1] >= lower - a[i]:
            L -= 1
        while R >= 0 and a[R] > upper - a[i]:
            R -= 1
        first = max(L, i + 1)
        add = R - first + 1 if R >= first else 0
        c += add
        steps.append(_step(f"i = {i} (a = {a[i]})", [("a", a, i)],
                           [("L", L, "indigo"), ("R", R, "violet"), ("+", add, "amber"),
                            ("count", c, "green")],
                           eq=f"partners need {lower - a[i]} <= a[j] <= {upper - a[i]}, j > {i}: "
                              f"indices {first}..{R} -> +{add}"))
    return _pack("Fair Pairs - one sweep, two boundaries that only move left",
                 "nums = [0, 1, 7, 4, 4, 5] (sorted), lower = 3, upper = 6", steps, c)


_ALL = None


def get(problem, form):
    """Return ``(title, input line, steps, final)`` for one solution form."""
    global _ALL
    if _ALL is None:
        _ALL = {
            ("two2", "ends"): _two_sum2(),
            ("container", "ends"): _container(),
            ("merge", "back"): _merge_back(),
            ("squares", "back"): _squares_back(),
            ("boats", "greedy"): _boats(),
            ("pairtopics", "count"): _pair_topics(),
            ("summary", "runs"): _summary_runs(),
            ("alt", "runs"): _alt_runs(),
            ("two2", "bs"): _two2_bs(),
            ("squares", "sort"): _squares_sort(),
            ("three", "bs"): _three_bs(),
            ("three", "squeeze"): _three_squeeze(),
            ("close", "bs"): _close_bs(),
            ("close", "squeeze"): _close_squeeze(),
            ("pairs", "bs"): _pairs_bs(),
            ("pairs", "count"): _pairs_count(),
            ("fair", "bs"): _fair_bs(),
            ("fair", "count"): _fair_count(),
            ("merge", "front"): _merge_front(),
            ("container", "brute"): _container_brute(),
            ("pairtopics", "bisect"): _pairs_bisect(),
            ("minpair", "ends"): _minpair(),
            ("fair", "sweep"): _fair_sweep(),
        }
    return _ALL[(problem, form)]
