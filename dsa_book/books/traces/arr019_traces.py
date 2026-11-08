"""Step boards for the contribution-counting chapter (sum over all subarrays, sum
over all pairs, counting subarrays with a property, complementary counting, fixing
the middle for triples, index-and-value conditions, small-value counting).

Drawing lives in ``dsa_viz.story``; ``get`` returns data only. Each story walks the
chapter's own official sample, at most eight frames, and every number comes from
running the algorithm - never from a hand-typed table. A frame shows whatever the
rung actually maintains: the window being summed, the per-index cover counts of a
contribution argument, the remainder histogram of a pair count, the sliding window
of an at-most counter, the prefix-XOR table that lets a middle index be fixed.

Every problem carries a ladder: a ``scan`` first frame set (the direct definition,
enumerating windows or pairs), a ``fast`` second set (the rung the pattern names)
and, where the pattern has one, a ``contest`` third set. ``get`` returns whichever
frame set the caller asks for, and every builder asserts its own answer against the
verified judge sample before returning.
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from trace_log import chip_step, line_step, row_step, show  # noqa: E402


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    if len(steps) > 8:                    # keep first, a spread, and the last frame
        keep = sorted({0, 1, 3, 5, 7, len(steps) - 2, len(steps) - 1})
        steps = [steps[i] for i in keep][:8]
    if not 1 <= len(steps) <= 8:
        raise AssertionError(f"{title}: {len(steps)} frames")
    return (title, inp, steps, text)


def _cells(vals):
    return [str(v) for v in vals]


def _window(vals, lo, hi):
    """The array with only the closed window lo..hi spelled out."""
    return [str(v) if lo <= k <= hi else "." for k, v in enumerate(vals)]


def _mask(vals, idx):
    return [str(v) if k == idx else "." for k, v in enumerate(vals)]


# ------------------------------------------------- LC 1588 sum of odd-length subarrays
OS_ARR = [1, 4, 2, 5, 3]
OS_ANS = 58


def _os_scan():
    total, steps = 0, []
    for length in (1, 3, 5):
        for i in range(len(OS_ARR) - length + 1):
            part = OS_ARR[i:i + length]
            total += sum(part)
            steps.append(row_step(
                f"window {i}..{i + length - 1} (length {length})",
                [("arr", _window(OS_ARR, i, i + length - 1), None),
                 ("sum", [sum(part)], None), ("total", [total], None)]))
    assert total == OS_ANS
    return _pack("Odd-length subarrays - enumerate every one", f"arr = {OS_ARR}", steps, total)


def _os_contrib():
    n, counts, total, steps = len(OS_ARR), [], 0, []
    for i, v in enumerate(OS_ARR):
        times = ((i + 1) * (n - i) + 1) // 2      # odd-length windows covering slot i
        counts.append(times)
        total += v * times
        steps.append(row_step(
            f"slot {i} holds {v}",
            [("arr", OS_ARR, i), ("times", counts, len(counts) - 1), ("total", [total], None)],
            eq=f"covers = (({i}+1)({n}-{i})+1)//2 = {times}"))
    assert total == OS_ANS
    return _pack("Contribution - charge each element once", f"arr = {OS_ARR}", steps, total)


# ------------------------------------------------------- LC 2104 sum of subarray ranges
RG_NUMS = [1, 2, 3]
RG_ANS = 4


def _rg_scan():
    total, steps = 0, []
    for i in range(len(RG_NUMS)):
        lo = hi = RG_NUMS[i]
        for j in range(i, len(RG_NUMS)):
            lo, hi = min(lo, RG_NUMS[j]), max(hi, RG_NUMS[j])
            total += hi - lo
            steps.append(row_step(
                f"subarray {i}..{j}",
                [("nums", _window(RG_NUMS, i, j), None),
                 ("range", [hi - lo], None), ("total", [total], None)],
                eq=f"{hi} - {lo}"))
    assert total == RG_ANS
    return _pack("Subarray ranges - two nested loops", f"nums = {RG_NUMS}", steps, total)


def _spans(nums, greater):
    """Left/right distances over which nums[i] stays the (max if greater else min).

    Strict on the left edge, non-strict on the right, so a run of equal values is
    charged to exactly one of its members."""
    n = len(nums)
    left, right, stack = [0] * n, [n] * n, []
    for i, v in enumerate(nums):
        while stack and (nums[stack[-1]] <= v if greater else nums[stack[-1]] >= v):
            stack.pop()
        left[i] = i - stack[-1] if stack else i + 1
        stack.append(i)
    stack = []
    for i in range(n - 1, -1, -1):
        v = nums[i]
        while stack and (nums[stack[-1]] < v if greater else nums[stack[-1]] > v):
            stack.pop()
        right[i] = stack[-1] - i if stack else n - i
        stack.append(i)
    return left, right


def _rg_stack():
    nums, n = RG_NUMS, len(RG_NUMS)
    mx_l, mx_r = _spans(nums, greater=True)
    mn_l, mn_r = _spans(nums, greater=False)
    smax = sum(v * mx_l[i] * mx_r[i] for i, v in enumerate(nums))
    smin = sum(v * mn_l[i] * mn_r[i] for i, v in enumerate(nums))
    steps = []
    for i, v in enumerate(nums):
        steps.append(row_step(
            f"slot {i}: {v} is the max of {mx_l[i] * mx_r[i]} windows",
            [("nums", nums, i), ("as max", [mx_l[i] * mx_r[i]], None),
             ("as min", [mn_l[i] * mn_r[i]], None)],
            eq=f"{v} x {mx_l[i]} x {mx_r[i]}"))
    steps.append(chip_step("subtract the two totals",
                           [("maxes", smax, "cyan"), ("mins", smin, "amber"),
                            ("answer", smax - smin, "green")],
                           eq="range = max - min, so sum ranges = sum maxes - sum mins"))
    assert smax - smin == RG_ANS
    return _pack("Charge max and min separately - one monotonic stack each",
                 f"nums = {nums}", steps, smax - smin)


# --------------------------------------------------------- LC 1512 number of good pairs
GP_NUMS = [1, 2, 3, 1, 1, 3]
GP_ANS = 4


def _gp_scan():
    ans, steps = 0, []
    for i in range(len(GP_NUMS)):
        for j in range(i + 1, len(GP_NUMS)):
            hit = GP_NUMS[i] == GP_NUMS[j]
            ans += hit
            steps.append(chip_step(
                f"pair ({i}, {j})", [("nums[i]", GP_NUMS[i], "indigo"),
                                      ("nums[j]", GP_NUMS[j], "pink"),
                                      ("good" if hit else "skip", ans, "green" if hit else "muted")]))
    assert ans == GP_ANS
    return _pack("Good pairs - test every (i, j)", f"nums = {GP_NUMS}", steps, ans)


def _gp_freq():
    seen, ans, steps = defaultdict(int), 0, []
    for i, v in enumerate(GP_NUMS):
        before = seen[v]
        ans += before
        seen[v] += 1
        steps.append(row_step(
            f"read {v} at slot {i}",
            [("nums", GP_NUMS, i),
             ("seen", [f"{k}:{n}" for k, n in sorted(seen.items())], None)],
            eq=f"{before} earlier copies -> {before} new pairs"))
    assert ans == GP_ANS
    return _pack("One pass - a new copy pairs with every earlier copy",
                 f"nums = {GP_NUMS}", steps, ans)


# --------------------------------- LC 1010 pairs of songs divisible by 60
SO_TIME = [30, 20, 150, 100, 40]
SO_ANS = 3


def _so_scan():
    ans, steps = 0, []
    for i in range(len(SO_TIME)):
        for j in range(i + 1, len(SO_TIME)):
            tot = SO_TIME[i] + SO_TIME[j]
            hit = tot % 60 == 0
            ans += hit
            steps.append(chip_step(f"pair ({i}, {j})",
                                   [("sum", tot, "indigo" if hit else "muted"),
                                    ("% 60", tot % 60, "amber"), ("answer", ans, "green")]))
    assert ans == SO_ANS
    return _pack("Song pairs - every pair, then take the modulo",
                 f"time = {SO_TIME}", steps, ans)


def _so_rem60():
    cnt, ans, steps = defaultdict(int), 0, []
    for i, t in enumerate(SO_TIME):
        r = t % 60
        need = (60 - r) % 60
        ans += cnt[need]
        cnt[r] += 1
        steps.append(row_step(
            f"song {t} -> remainder {r}",
            [("time", SO_TIME, i),
             ("hist", [f"{k}:{n}" for k, n in sorted(cnt.items()) if n], None),
             ("need", [need], None)],
            eq=f"complement {need} seen {cnt[need] - (1 if need == r else 0)} time(s) -> answer {ans}"))
    assert ans == SO_ANS
    return _pack("Only remainders matter - 60 buckets", f"time = {SO_TIME}", steps, ans)


# -------------------------------------------------- LC 930 binary subarrays with sum
BS_NUMS = [1, 0, 1, 0, 1]
BS_GOAL = 2
BS_ANS = 4


def _bs_scan():
    ans, steps = 0, []
    for i in range(len(BS_NUMS)):
        s = 0
        for j in range(i, len(BS_NUMS)):
            s += BS_NUMS[j]
            hit = s == BS_GOAL
            ans += hit
            steps.append(row_step(
                f"window {i}..{j}",
                [("nums", _window(BS_NUMS, i, j), None), ("sum", [s], None),
                 ("== goal", [int(hit)], None), ("answer", [ans], None)]))
    assert ans == BS_ANS
    return _pack("Subarrays with sum goal - enumerate all of them",
                 f"nums = {BS_NUMS}, goal = {BS_GOAL}", steps, ans)


def _atmost(nums, goal):
    """Number of subarrays with sum <= goal, for a binary array (sliding window)."""
    if goal < 0:
        return 0
    left, total, ans = 0, 0, 0
    for right, v in enumerate(nums):
        total += v
        while total > goal:
            total -= nums[left]
            left += 1
        ans += right - left + 1        # every start in left..right keeps the sum <= goal
    return ans


def _bs_atmost():
    nums, goal = BS_NUMS, BS_GOAL
    steps, left, total = [], 0, 0
    hi = 0
    for right, v in enumerate(nums):
        total += v
        while total > goal:
            total -= nums[left]
            left += 1
        hi += right - left + 1
        steps.append(row_step(
            f"right = {right}",
            [("nums", _window(nums, left, right), None),
             ("left", [left], None), ("sum", [total], None),
             ("count", [hi], None)],
            eq=f"starts {left}..{right} = {right - left + 1} subarrays"))
    a = _atmost(nums, goal)
    b = _atmost(nums, goal - 1)
    assert (a, b, a - b) == (hi, hi - BS_ANS, BS_ANS)
    steps.append(chip_step("exactly = at most - at minus one",
                           [("at most 2", a, "cyan"), ("at most 1", b, "amber"),
                            ("exactly 2", a - b, "green")], eq=f"{a} - {b} = {a - b}"))
    return _pack("Exactly goal = at most goal - at most (goal-1)",
                 f"nums = {nums}, goal = {goal}", steps, a - b)


def _bs_prefix():
    nums, goal = BS_NUMS, BS_GOAL
    cnt, pre, ans, steps = defaultdict(int), 0, 0, []
    cnt[0] = 1
    for i, v in enumerate(nums):
        pre += v
        ans += cnt.get(pre - goal, 0)    # .get: a lookup must not insert a key
        cnt[pre] += 1
        steps.append(row_step(
            f"prefix after slot {i}",
            [("nums", nums, i), ("prefix", [pre], None),
             ("needs", [pre - goal], None),
             ("table", [f"{k}:{n}" for k, n in sorted(cnt.items())], None),
             ("answer", [ans], None)],
            eq=f"earlier prefixes equal to {pre - goal}: {cnt.get(pre - goal, 0)}"))
    assert ans == BS_ANS
    return _pack("Prefix sums counted by a hash map",
                 f"nums = {nums}, goal = {goal}", steps, ans)


# ------------------------------------------------ LC 1513 substrings with only 1s
ON_S = "0110111"
ON_ANS = 9


def _on_runs():
    runs, i, steps = [], 0, []
    s = ON_S
    total = 0
    while i < len(s):
        if s[i] == "1":
            j = i
            while j < len(s) and s[j] == "1":
                j += 1
            runs.append(j - i)
            total += (j - i) * (j - i + 1) // 2
            steps.append(row_step(
                f"run at {i}..{j - 1} has length {j - i}",
                [("s", list(s), None), ("hi", _window([1] * len(s), i, j - 1), None),
                 ("run", [(j - i) * (j - i + 1) // 2], None),
                 ("answer", [total], None)],
                eq=f"L(L+1)/2 with L = {j - i}"))
            i = j
        else:
            i += 1
    assert total == ON_ANS
    return _pack("Ones - split into runs, close each with a triangle number",
                 f's = "{s}"', steps, total)


def _on_stream():
    k, ans, steps = 0, 0, []
    for i, c in enumerate(ON_S):
        k = 0 if c == "0" else k + 1
        ans += k
        steps.append(row_step(
            f"read '{c}' at {i}",
            [("s", list(ON_S), i), ("run", [k], None), ("answer", [ans], None)],
            eq=f"{k} new all-ones substrings end here"))
    assert ans == ON_ANS
    return _pack("Ones - one pass, count the substrings that END here",
                 f's = "{ON_S}"', steps, ans)


REGISTRY = {
    ("oddsum", "scan"): _os_scan(),
    ("oddsum", "contrib"): _os_contrib(),
    ("ranges", "scan"): _rg_scan(),
    ("ranges", "stack"): _rg_stack(),
    ("goodpairs", "scan"): _gp_scan(),
    ("goodpairs", "freq"): _gp_freq(),
    ("songs", "scan"): _so_scan(),
    ("songs", "rem60"): _so_rem60(),
    ("binsum", "scan"): _bs_scan(),
    ("binsum", "atmost"): _bs_atmost(),
    ("binsum", "prefix"): _bs_prefix(),
    ("ones", "runs"): _on_runs(),
    ("ones", "stream"): _on_stream(),
}


def get(problem, form):
    return REGISTRY[(problem, form)]


# ------------------------------------------------------------------ the middle rungs
# Each problem's ladder needs three storyboards, so the middle rung of every pattern
# (the one that is neither the definition nor the final form) is built here. The boards
# are registered at the foot of the module, once every sample constant exists.


def _os_roll():
    """LC 1588 rung 2: a prefix-sum array turns each window sum into one subtraction."""
    px = [0]
    for v in OS_ARR:
        px.append(px[-1] + v)
    total, steps = 0, []
    for length in (1, 3, 5):
        parts = [px[i + length] - px[i] for i in range(len(OS_ARR) - length + 1)]
        total += sum(parts)
        steps.append(row_step(
            f"all windows of length {length}",
            [("arr", OS_ARR, None), ("sums", parts, None),
             ("total", [total], None)],
            eq=f"{length} window(s): {' + '.join(str(p) for p in parts)} -> total {total}"))
    assert total == OS_ANS
    return _pack("Odd-length subarrays - prefix sums, one subtraction per window",
                 f"arr = {OS_ARR}", steps, total)


def _rg_naive():
    """LC 2104 rung 1: re-scan the whole window for its min and max, every time."""
    nums = RG_NUMS
    total, steps = 0, []
    for i in range(len(nums)):
        for j in range(i, len(nums)):
            part = nums[i:j + 1]
            gap = max(part) - min(part)
            total += gap
            steps.append(row_step(
                f"subarray {i}..{j}: rescan all {len(part)} cells",
                [("nums", _window(nums, i, j), None), ("range", [gap], None),
                 ("total", [total], None)],
                eq=f"max {max(part)} - min {min(part)} = {gap}"))
    assert total == RG_ANS
    return _pack("Subarray ranges - min and max recomputed from scratch",
                 f"nums = {nums}", steps, total)


def _gp_tri():
    """LC 1512 rung 3: count first, then each value's copies contribute C(c, 2)."""
    cnt = Counter(GP_NUMS)
    ans, steps = 0, []
    for v in sorted(cnt):
        c = cnt[v]
        ans += c * (c - 1) // 2
        steps.append(row_step(
            f"value {v} appears {c} times",
            [("nums", [str(x) if x == v else "." for x in GP_NUMS], None),
             ("count", [c], None), ("pairs", [c * (c - 1) // 2], None),
             ("answer", [ans], None)],
            eq=f"C({c}, 2) = {c}({c}-1)/2 = {c * (c - 1) // 2}"))
    assert ans == GP_ANS
    return _pack("Good pairs - one count per value, then a triangle number",
                 f"nums = {GP_NUMS}", steps, ans)


def _so_hist():
    """LC 1010 rung 3: fill the 60-bucket histogram, then match buckets to complements."""
    cnt = Counter(t % 60 for t in SO_TIME)
    ans, steps = 0, []
    for r in sorted(cnt):
        c = cnt[r]
        if r == 0 or r == 30:                     # its own complement
            add = c * (c - 1) // 2
        elif r > 30:                              # already paired with 60 - r
            continue
        else:
            add = c * cnt.get(60 - r, 0)
        ans += add
        steps.append(row_step(
            f"remainder {r} holds {cnt[r]} song(s)",
            [("time", [str(t) if t % 60 == r else "." for t in SO_TIME], None),
             ("pairs", [add], None), ("answer", [ans], None)],
            eq=(f"C({cnt[r]}, 2) = {add} (self-complement)" if r in (0, 30) else
                f"{cnt[r]} x {cnt.get(60 - r, 0)} songs at remainder {60 - r}")))
    assert ans == SO_ANS
    return _pack("Song pairs - pair whole buckets at once", f"time = {SO_TIME}", steps, ans)


def _on_scan():
    """LC 1513 rung 1: test every substring for 'all ones'."""
    s, ans, steps = ON_S, 0, []
    for i in range(len(s)):
        for j in range(i, len(s)):
            hit = "0" not in s[i:j + 1]
            ans += hit
            steps.append(row_step(
                f"substring s[{i}..{j}] = '{s[i:j + 1]}'",
                [("s", _window(list(s), i, j), None), ("all 1s", [int(hit)], None),
                 ("answer", [ans], None)],
                eq=f"length {j - i + 1}"))
    assert ans == ON_ANS
    return _pack("All-ones substrings - enumerate and test", f's = "{s}"', steps, ans)


def _lk_sortruns():
    """LC 1394 rung 2: sort, and a run's length is its frequency for free."""
    arr = sorted(LUCK_ARR)
    ans, steps, i = -1, [], 0
    while i < len(arr):
        j = i
        while j < len(arr) and arr[j] == arr[i]:
            j += 1
        length = j - i
        if length == arr[i]:
            ans = max(ans, arr[i])
        steps.append(row_step(
            f"run of {arr[i]} spans {i}..{j - 1}",
            [("sorted", _window(arr, i, j - 1), None), ("length", [length], None),
             ("best", [ans], None)],
            eq=f"length {length} {'==' if length == arr[i] else '!='} value {arr[i]}"))
        i = j
    assert ans == LUCK_ANS
    return _pack("Lucky integer - sort and measure the runs", f"arr = {LUCK_ARR}", steps, ans)


def _gt_walk():
    """LC 1534 rung 2: fix the middle, but sum the value interval cell by cell."""
    arr, n = GT_ARR, len(GT_ARR)
    pc = [0] * (GT_MAXV + 1)
    pc[arr[0]] = 1
    ans, steps = 0, []
    for j in range(1, n - 1):
        for k in range(j + 1, n):
            if abs(arr[j] - arr[k]) > GT_B:
                continue
            lo = max(0, arr[j] - GT_A, arr[k] - GT_C)
            hi = min(GT_MAXV, arr[j] + GT_A, arr[k] + GT_C)
            if lo > hi:
                continue
            cells = [pc[v] for v in range(lo, hi + 1)]
            hit = sum(pc[lo:hi + 1])
            ans += hit
            steps.append(row_step(
                f"middle j = {j}, right end k = {k}: walk values {lo}..{hi}",
                [("arr", arr, j), ("pc[v]", cells, None),
                 ("answer", [ans], None)],
                eq=f"walk {hi} - {lo} + 1 = {hi - lo + 1} buckets -> {hit} match(es)"))
        pc[arr[j]] += 1
    assert ans == GT_ANS
    return _pack("Fix the middle - the interval is still walked one cell at a time",
                 f"arr = {GT_ARR}, a={GT_A}, b={GT_B}, c={GT_C}", steps, ans)


def _by_ff():
    """LC 1781 rung 3: count the *frequencies* so max and min are cursor moves."""
    s = BY_S
    ans, steps = 0, []
    for i in range(len(s)):
        cnt = [0] * 26
        how = [0] * (len(s) + 2)        # how[f] = how many letters sit at frequency f
        nz = hi = lo = 0
        row = []
        for j in range(i, len(s)):
            c = ord(s[j]) - 97
            how[cnt[c]] -= 1
            if cnt[c] == 0:
                nz += 1
                hi = max(hi, 1) if hi == 0 else hi
            cnt[c] += 1
            how[cnt[c]] += 1
            hi = max(hi, cnt[c])
            while how[hi] == 0:
                hi -= 1
            lo = 1 if how[1] else next(f for f in range(1, hi + 1) if how[f])
            row.append(hi - lo)
        ans += sum(row)
        steps.append(row_step(
            f"extend from left {i}, cursors only move right (then fall back)",
            [("s", _window(list(s), i, len(s) - 1), None), ("beauty", row, None),
             ("answer", [ans], None)],
            eq=f"letters seen {nz}, final frequency span {hi} - {lo} -> "
               f"beauties {row}"))
    assert ans == BY_ANS
    return _pack("Substring beauty - a counter of counters for max and min",
                 f's = "{s}"', steps, ans)


# ---------------------------------------------------------------- LC 1534 count good triplets
GT_ARR = [3, 0, 1, 1, 9, 7]
GT_A, GT_B, GT_C = 7, 2, 3
GT_ANS = 4
GT_MAXV = 1000                        # constraints: 0 <= arr[i] <= 1000


def _gt_scan():
    ans, steps = 0, []
    for i in range(len(GT_ARR)):
        for j in range(i + 1, len(GT_ARR)):
            for k in range(j + 1, len(GT_ARR)):
                ok = (abs(GT_ARR[i] - GT_ARR[j]) <= GT_A
                      and abs(GT_ARR[j] - GT_ARR[k]) <= GT_B
                      and abs(GT_ARR[i] - GT_ARR[k]) <= GT_C)
                ans += ok
                steps.append(chip_step(f"triplet i={i}, j={j}, k={k}",
                                       [("arr[i]", GT_ARR[i], "indigo"),
                                        ("arr[j]", GT_ARR[j], "cyan"),
                                        ("arr[k]", GT_ARR[k], "pink"),
                                        ("counted" if ok else "rejected", ans,
                                         "green" if ok else "muted")],
                                       eq=", ".join(
                                           f"|{x}-{y}|={abs(x - y)}{'<=' if abs(x - y) <= c else '>'}{c}"
                                           for x, y, c in ((GT_ARR[i], GT_ARR[j], GT_A),
                                                           (GT_ARR[j], GT_ARR[k], GT_B),
                                                           (GT_ARR[i], GT_ARR[k], GT_C)))))
    assert ans == GT_ANS
    return _pack("Good triplets - test every (i, j, k)", f"arr = {GT_ARR}", steps, ans)


def _gt_middle():
    """Fix the middle j; the two remaining conditions become one value interval."""
    arr, n = GT_ARR, len(GT_ARR)
    pc = [0] * (GT_MAXV + 1)           # pc[v] = how many indices i < j hold the value v
    pc[arr[0]] = 1                     # slot 0 is already to the left of every middle we visit
    ans, steps = 0, []
    for j in range(1, n - 1):
        cum = [0] * (GT_MAXV + 2)
        for v in range(GT_MAXV + 1):
            cum[v + 1] = cum[v] + pc[v]    # cum[x+1] = number of i < j with arr[i] <= x
        for k in range(j + 1, n):
            if abs(arr[j] - arr[k]) > GT_B:
                continue
            lo = max(0, arr[j] - GT_A, arr[k] - GT_C)
            hi = min(GT_MAXV, arr[j] + GT_A, arr[k] + GT_C)
            if lo > hi:
                continue
            hit = cum[hi + 1] - cum[lo]
            ans += hit
            steps.append(row_step(
                f"middle j = {j}, right end k = {k}: arr[i] in [{lo}, {hi}]",
                [("arr", arr, j), ("pool", [f"{v}:{pc[v]}" for v in range(GT_MAXV + 1)
                                                 if pc[v]] or ["empty"], None),
                 ("hits", [hit], None), ("answer", [ans], None)],
                eq=f"|{arr[j]}-{arr[k]}| = {abs(arr[j] - arr[k])} <= {GT_B}, so "
                   f"{hit} earlier value(s) satisfy both remaining bounds"))
        pc[arr[j]] += 1                # j becomes part of the left pool for the next middle
    assert ans == GT_ANS
    return _pack("Fix the middle - a value histogram answers the left side in O(1)",
                 f"arr = {GT_ARR}, a={GT_A}, b={GT_B}, c={GT_C}", steps, ans)


# ------------------------------------------------- LC 1442 triplets with equal XOR halves
XOR_ARR = [2, 3, 1, 6, 7]
XOR_ANS = 4


def _px(arr):
    out = [0]
    for v in arr:
        out.append(out[-1] ^ v)
    return out


def _xt_scan():
    arr, n = XOR_ARR, len(XOR_ARR)
    ans, steps = 0, []
    for i in range(n - 1):
        for j in range(i + 1, n):
            for k in range(j, n):
                a = b = 0
                for v in arr[i:j]:
                    a ^= v
                for v in arr[j:k + 1]:
                    b ^= v
                ans += a == b
                steps.append(chip_step(f"(i, j, k) = ({i}, {j}, {k})",
                                       [("a", a, "indigo"), ("b", b, "cyan"),
                                        ("answer", ans, "green" if a == b else "muted")],
                                       eq=f"xor of {arr[i:j]} = {a}, xor of {arr[j:k + 1]} = {b}"))
    assert ans == XOR_ANS
    return _pack("Equal-XOR triplets - recompute both halves every time",
                 f"arr = {XOR_ARR}", steps, ans)


def _xt_prefix():
    arr, n = XOR_ARR, len(XOR_ARR)
    px = _px(arr)
    ans, steps = 0, []
    for i in range(n - 1):
        wins = [k for k in range(i, n) if px[k + 1] == px[i]]
        gain = sum(k - i for k in wins)
        ans += gain
        steps.append(row_step(
            f"left end i = {i}",
            [("px", px, i), ("k", [str(k) for k in wins] or ["-"], None),
             ("answer", [ans], None)],
            eq=(f"arr[i..k] has XOR 0 for k in {wins}; each offers k-i="
                f"{[k - i for k in wins]} choices of j" if wins else
                f"no k has px[k+1] == px[{i}] == {px[i]}")))
    assert ans == XOR_ANS
    return _pack("Zero-xor spans - the middle index is free", f"arr = {XOR_ARR}", steps, ans)


def _xt_buckets():
    px = _px(XOR_ARR)
    cnt, tot, ans, steps = defaultdict(int), defaultdict(int), 0, []
    for t, x in enumerate(px):
        new = t * cnt[x] - tot[x] - cnt[x]
        ans += new
        cnt[x] += 1
        tot[x] += t
        steps.append(row_step(
            f"prefix xor {x} at index {t}",
            [("px", px, t),
             ("count", [f"{k}:{cnt[k]}" for k in sorted(cnt)], None),
             ("idxsum", [f"{k}:{tot[k]}" for k in sorted(cnt)], None),
             ("answer", [ans], None)],
            eq=f"{new} new triplets from {cnt[x] - 1} earlier copies of {x}"))
    assert ans == XOR_ANS
    return _pack("One pass - count and index-sum per prefix value", f"arr = {XOR_ARR}",
                 steps, ans)


# -------------------------------------------------------- LC 1394 lucky integer (value == frequency)
LUCK_ARR = [1, 2, 2, 3, 3, 3]
LUCK_ANS = 3


def _lk_scan():
    ans, steps = -1, []
    for v in sorted(set(LUCK_ARR)):
        f = LUCK_ARR.count(v)
        if f == v:
            ans = max(ans, v)
        steps.append(row_step(
            f"value {v}: one full scan per candidate",
            [("arr", LUCK_ARR, None), ("freq", [f], None), ("best", [ans], None)],
            eq=f"count({v}) = {f} {'==' if f == v else '!='} {v}"))
    assert ans == LUCK_ANS
    return _pack("Lucky integer - test each distinct value with a scan",
                 f"arr = {LUCK_ARR}", steps, ans)


def _lk_direct():
    cnt = [0] * 501
    ans, steps = -1, []
    for i, v in enumerate(LUCK_ARR):
        cnt[v] += 1
        steps.append(row_step(
            f"read {v} at slot {i}",
            [("arr", LUCK_ARR, i),
             ("table", [f"{x}:{cnt[x]}" for x in range(1, 501) if cnt[x]], None),
             ("best", [ans], None)],
            eq=f"bucket {v} now holds {cnt[v]}"))
    for v in range(1, 501):
        if cnt[v] == v:
            ans = max(ans, v)
    steps.append(chip_step("sweep the table once counting is done",
                           [("largest v with cnt[v] == v", ans, "green")],
                           eq="a bucket can only be judged after its value stops changing"))
    assert ans == LUCK_ANS
    return _pack("Lucky integer - direct addressing by value", f"arr = {LUCK_ARR}", steps, ans)


# ------------------------------------------------------- LC 1781 sum of beauty of substrings
BY_S = "aabcb"
BY_ANS = 5


def _by_scan():
    s = BY_S
    ans, steps = 0, []
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            sub = s[i:j]
            freqs = [sub.count(ch) for ch in set(sub)]
            b = max(freqs) - min(freqs)
            ans += b
            steps.append(row_step(
                f"substring s[{i}..{j - 1}] = '{sub}'",
                [("s", _window(list(s), i, j - 1), None), ("beauty", [b], None),
                 ("answer", [ans], None)],
                eq=f"{max(freqs)} - {min(freqs)} = {b}"))
    assert ans == BY_ANS
    return _pack("Substring beauty - recount every substring from scratch",
                 f's = "{s}"', steps, ans)


def _by_extend():
    s = BY_S
    ans, steps = 0, []
    for i in range(len(s)):
        cnt = [0] * 26
        row = []
        for j in range(i, len(s)):
            cnt[ord(s[j]) - 97] += 1
            row.append(max(cnt) - min(c for c in cnt if c))
        ans += sum(row)
        steps.append(row_step(
            f"extend the right end from left {i}",
            [("s", _window(list(s), i, len(s) - 1), None), ("beauty", row, None),
             ("answer", [ans], None)],
            eq=f"'{s[i:]}' -> {row}, +{sum(row)}"))
    assert ans == BY_ANS
    return _pack("Substring beauty - carry the histogram as the right end moves",
                 f's = "{s}"', steps, ans)


REGISTRY.update({
    ("triplets", "scan"): _gt_scan(),
    ("triplets", "middle"): _gt_middle(),
    ("xortri", "scan"): _xt_scan(),
    ("xortri", "prefix"): _xt_prefix(),
    ("xortri", "buckets"): _xt_buckets(),
    ("lucky", "scan"): _lk_scan(),
    ("lucky", "direct"): _lk_direct(),
    ("beauty", "scan"): _by_scan(),
    ("beauty", "extend"): _by_extend(),
})

# The middle rungs of the ladder, built above but registered last so that every
# sample constant already exists.
REGISTRY.update({
    ("oddsum", "roll"): _os_roll(),
    ("ranges", "naive"): _rg_naive(),
    ("goodpairs", "tri"): _gp_tri(),
    ("songs", "hist"): _so_hist(),
    ("ones", "scan"): _on_scan(),
    ("triplets", "walk"): _gt_walk(),
    ("lucky", "sortruns"): _lk_sortruns(),
    ("beauty", "ff"): _by_ff(),
})
