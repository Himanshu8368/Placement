"""Step frames for the partitioning & in-place rearrangement chapter.

Drawing lives in ``dsa_viz.story``. ``get`` returns data only. Each story
walks the chapter's own sample, at most eight frames, and the numbers come
from running the algorithm - never from a hand-typed table.
A partition frame shows the array with the current cursor (or the pair of
cursors) highlighted plus the rolling chips (lo, mid, hi, i, j, buckets),
so the reader watches a *swap* or a *write* at a boundary exactly as the
algorithm makes it.

Every problem carries the full ladder: a ``brute`` frame set (the direct
definition, often a nested scan or a split-and-refill), an ``intermediate``
frame set (a sort-based or two-pointer view that gets the shape right but
not yet the O(1) insight, or a wrong-greedy / unstable counterexample),
and an ``optimal`` frame set (the rung the pattern names). ``get`` returns
whichever frame set the caller asks for.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from trace_log import chip_step, row_step, show  # noqa: E402


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


def _row(name, cells, hi=None):
    """One array row. ``hi`` is a single integer index (green highlight)."""
    return (name, [str(c) for c in cells], hi)


# ------------------------------------------------------- LC 2161 partition pivot

_PIVOT_NUMS = [9, 12, 3, 5, 14, 10, 7, 6, 21]
_PIVOT = 10


def _pivot_brute():
    """Three nested passes, one per bucket. O(n) time, O(n) space, stable."""
    nums, p = list(_PIVOT_NUMS), _PIVOT
    steps = [_step("input", [_row("nums", nums)],
                   [("<", 0, "indigo"), ("=", 0, "cyan"), (">", 0, "amber")],
                   eq=f"pass 1: copy every x < {p}; pass 2: every x == {p}; pass 3: every x > {p}")]
    res = []
    for tag, keep in (("less", lambda x: x < p),
                       ("equal", lambda x: x == p),
                       ("greater", lambda x: x > p)):
        for x in nums:
            if keep(x):
                res.append(x)
        steps.append(_step(f"pass {tag}",
                           [_row("nums", nums), _row("res", res + ["."] * (len(nums) - len(res)),
                                                     hi=len(res) - 1 if res else None)],
                           eq=f"after {tag}-pass: res={res}"))
    assert res == [9, 3, 5, 7, 6, 10, 12, 14, 21]
    return _pack("Partition Array - three-pass (oracle)",
                 f"nums = {nums}, pivot = {p}", steps, str(res))


def _pivot_threelist():
    """Single scan into three lists; concat at the end. Optimal stable."""
    nums, p = list(_PIVOT_NUMS), _PIVOT
    less, equal, greater = [], [], []
    steps = [_step("input", [_row("nums", nums)],
                   [("i", 0, "indigo")],
                   eq="one scan: bucket each x by comparing to pivot")]
    for i, x in enumerate(nums):
        if x < p:
            less.append(x)
            act = f"less += {x}"
        elif x == p:
            equal.append(x)
            act = f"equal += {x}"
        else:
            greater.append(x)
            act = f"greater += {x}"
        steps.append(_step(f"see {x} at {i}",
                           [_row("nums", nums, hi=i), _row("less", less),
                            _row("equal", equal), _row("greater", greater)],
                           [("i", i, "indigo")],
                           eq=act))
    res = less + equal + greater
    if len(steps) > 8:
        steps = [steps[0]] + steps[-7:]
    assert res == [9, 3, 5, 7, 6, 10, 12, 14, 21]
    return _pack("Partition Array - single-scan three lists (stable)",
                 f"nums = {nums}, pivot = {p}", steps, str(res))


def _pivot_twoend():
    """One buffer, two write cursors: < p from the front, > p from the back, p in the gap."""
    nums, p = list(_PIVOT_NUMS), _PIVOT
    n = len(nums)
    res = ["."] * n
    lo, hi = 0, n - 1
    steps = [_step("input", [_row("nums", nums), _row("res", res)],
                   [("lo", lo, "green"), ("hi", hi, "amber")],
                   eq="step i reads nums[i] (front scan) and nums[n-1-i] (back scan)")]
    shown = {0, 2, 4, 6, 8}
    for i in range(n):
        x, y = nums[i], nums[n - 1 - i]
        did = []
        if x < p:
            res[lo] = x
            did.append(f"res[{lo}] = {x}")
            lo += 1
        if y > p:
            res[hi] = y
            did.append(f"res[{hi}] = {y}")
            hi -= 1
        if i in shown:
            steps.append(_step(f"i = {i}", [_row("nums", nums, hi=i), _row("res", res)],
                               [("lo", lo, "green"), ("hi", hi, "amber")],
                               eq=(", ".join(did) or "nothing to write") +
                               f"  (front reads {x}, back reads {y})"))
    for k in range(lo, hi + 1):
        res[k] = p
    steps.append(_step("fill the gap", [_row("res", res, hi=lo)],
                       [("lo", lo, "green"), ("hi", hi, "amber")],
                       eq=f"res[{lo}..{hi}] = {p}: the copies of the pivot"))
    assert res == [9, 3, 5, 7, 6, 10, 12, 14, 21]
    return _pack("Partition Array - two write cursors, one buffer",
                 f"nums = {nums}, pivot = {p}", steps, str(res))


def _pivot_dnf():
    """In-place Dutch National Flag. UNSTABLE - shows the failure mode."""
    a = list(_PIVOT_NUMS)
    n = len(a)
    lo, mid, hi = 0, 0, n - 1
    p = _PIVOT
    steps = [_step("input", [_row("a", a)],
                   [("lo", lo, "green"), ("mid", mid, "indigo"), ("hi", hi, "amber")],
                   eq="regions: a[0..lo) < p; a[lo..mid) == p; a[mid..hi] unknown; a(hi..n) > p")]
    while mid <= hi:
        if a[mid] < p:
            a[lo], a[mid] = a[mid], a[lo]
            lo += 1
            mid += 1
            note = f"{a[mid - 1]} < p, swap(lo, mid); both ++"
        elif a[mid] == p:
            mid += 1
            note = f"== p, mid++"
        else:
            a[mid], a[hi] = a[hi], a[mid]
            hi -= 1
            note = f"> p, swap(mid, hi); hi-- (mid stays: incoming value unknown)"
        steps.append(_step(note,
                           [_row("a", a, hi=mid - 1 if mid > 0 else 0)],
                           [("lo", lo, "green"), ("mid", mid, "indigo"), ("hi", hi, "amber")],
                           eq=note))
    if len(steps) > 8:
        steps = [steps[0]] + steps[-7:]
    assert a[:len(a)].count(p) == 1  # sanity, still contains pivot once
    return _pack("Partition Array - in-place DNF (unstable)",
                 f"nums = {_PIVOT_NUMS}, pivot = {p}", steps, str(a))


# --------------------------------------------------- LC 2149 rearrange by sign

_SIGNS_NUMS = [3, 1, -2, -5, 2, -4]


def _signs_brute():
    """Split into pos/neg lists, zip into answer. O(n), stable."""
    nums = list(_SIGNS_NUMS)
    pos = [x for x in nums if x > 0]
    neg = [x for x in nums if x < 0]
    steps = [_step("input", [_row("nums", nums)],
                   eq="pos = positives in order; neg = negatives in order; alternate")]
    res = []
    for i in range(len(pos)):
        res.append(pos[i])
        res.append(neg[i])
        steps.append(_step(f"pair {i}",
                           [_row("nums", nums), _row("pos", pos), _row("neg", neg),
                            _row("res", res, hi=len(res) - 1)],
                           eq=f"res[{2 * i}]={pos[i]}, res[{2 * i + 1}]={neg[i]}"))
    if len(steps) > 8:
        steps = [steps[0]] + steps[-7:]
    assert res == [3, -2, 1, -5, 2, -4]
    return _pack("Rearrange by Sign - split + zip (stable)",
                 f"nums = {nums}", steps, str(res))


def _signs_twoptr():
    """Result-array write with two cursors over the input. Stable, O(n)."""
    nums = list(_SIGNS_NUMS)
    n = len(nums)
    res = [0] * n
    pc = 0  # next positive source
    nc = 0  # next negative source
    steps = [_step("input", [_row("nums", nums), _row("res", ["?"] * n)],
                   [("even", 0, "indigo"), ("odd", 1, "amber")],
                   eq="walk result by twos; even slots get positives, odd slots get negatives")]
    for i in range(0, n, 2):
        while nums[pc] < 0:
            pc += 1
        while nums[nc] > 0:
            nc += 1
        res[i] = nums[pc]
        res[i + 1] = nums[nc]
        pc += 1
        nc += 1
        steps.append(_step(f"fill res[{i}], res[{i + 1}]",
                           [_row("nums", nums), _row("res", res, hi=i + 1)],
                           [("even", i, "indigo"), ("odd", i + 1, "amber")],
                           eq=f"res[{i}]={res[i]}, res[{i+1}]={res[i+1]}"))
    assert res == [3, -2, 1, -5, 2, -4]
    return _pack("Rearrange by Sign - result-array two-cursor",
                 f"nums = {nums}", steps, str(res))


def _signs_writecursors():
    """One read pass, two write cursors: positives to 0,2,4,..., negatives to 1,3,5,..."""
    nums = list(_SIGNS_NUMS)
    n = len(nums)
    res = ["?"] * n
    pw, nw = 0, 1
    steps = [_step("input", [_row("nums", nums), _row("res", res)],
                   [("pw", pw, "indigo"), ("nw", nw, "amber")],
                   eq="read nums once; a positive goes to res[pw], pw += 2; a negative to res[nw], nw += 2")]
    for i, x in enumerate(nums):
        if x > 0:
            res[pw] = x
            at = pw
            pw += 2
        else:
            res[nw] = x
            at = nw
            nw += 2
        steps.append(_step(f"read nums[{i}] = {x}", [_row("nums", nums, hi=i), _row("res", res, hi=at)],
                           [("pw", pw, "indigo"), ("nw", nw, "amber")],
                           eq=f"res[{at}] = {x}"))
    assert res == [3, -2, 1, -5, 2, -4]
    return _pack("Rearrange by Sign - one pass, two write cursors",
                 f"nums = {nums}", steps, str(res))


def _signs_inplace():
    """Unstable two-cursor partition to [pos | neg], then rebuild via halves.

    Shows the shape of the O(1)-space approach; the two-cursor partition is
    the same idea as QuickSort's outer Hoare step, and it deliberately loses
    stability (LC 2149 wants the stable answer, so this rung is the trade-off
    illustration rather than a submission)."""
    a = [-1, 2, -3, 4, -5, 6]
    n = len(a)
    orig = list(a)
    steps = [_step("input", [_row("a", a)],
                   eq="phase 1: two-cursor partition - positives to the front, negatives to the back")]
    lo, hi = 0, n - 1
    while lo < hi:
        while lo < hi and a[lo] > 0:
            lo += 1
        while lo < hi and a[hi] < 0:
            hi -= 1
        if lo < hi:
            a[lo], a[hi] = a[hi], a[lo]
            lo += 1
            hi -= 1
            steps.append(_step(f"swap at lo={lo-1}, hi={hi+1}",
                                [_row("a", a, hi=lo - 1)],
                                [("lo", lo, "indigo"), ("hi", hi, "amber")],
                                eq="both cursors landed on a violation -> swap; both advance"))
    half = n // 2
    pos = a[:half]
    neg = a[half:]
    out = []
    for i in range(half):
        out.append(pos[i])
        out.append(neg[i])
    steps.append(_step("after partition", [_row("pos", pos), _row("neg", neg)],
                        eq=f"pos = a[0..{half}) = {pos}; neg = a[{half}..{n}) = {neg}"))
    steps.append(_step("zip halves", [_row("res", out, hi=n - 1)],
                        eq="alternate pos[0], neg[0], pos[1], neg[1], ..."))
    # The partition order is unstable: 6 was at the back of the input but ends
    # up at the front of pos. That is the whole point of showing this rung.
    assert out[0] > 0 and out[1] < 0
    assert sorted(out[0::2]) == [2, 4, 6] and sorted(out[1::2]) == [-5, -3, -1]
    return _pack("Rearrange by Sign - in-place (unstable, two-cursor partition)",
                 f"a = {orig}", steps, str(out))


# ------------------------------------------------- LC 922 sort array parity II

_PARITY_NUMS = [4, 2, 5, 7]


def _parity_brute():
    """Split into evens/odds; refill result[i] by parity."""
    nums = list(_PARITY_NUMS)
    evens = [x for x in nums if x % 2 == 0]
    odds = [x for x in nums if x % 2 == 1]
    res = [0] * len(nums)
    steps = [_step("input", [_row("nums", nums)],
               eq="evens = even values; odds = odd values; even slots get evens, odd slots get odds")]
    ei = oi = 0
    for i in range(len(nums)):
        if i % 2 == 0:
            res[i] = evens[ei]
            ei += 1
        else:
            res[i] = odds[oi]
            oi += 1
        steps.append(_step(f"fill res[{i}]",
                           [_row("nums", nums), _row("evens", evens), _row("odds", odds),
                            _row("res", res, hi=i)],
                           eq=f"res[{i}]={res[i]}"))
    if len(steps) > 8:
        steps = [steps[0]] + steps[-7:]
    assert res == [4, 5, 2, 7]
    return _pack("Sort Array by Parity II - split + refill",
                 f"nums = {nums}", steps, str(res))


def _parity_pairscan():
    """i steps by 2 on even slots; for each violation, scan j from 1 by 2."""
    a = list(_PARITY_NUMS)
    n = len(a)
    steps = [_step("input", [_row("a", a)],
                   [("i", 0, "indigo"), ("j", 1, "amber")],
                   eq="for each even index i, if a[i] is odd find an odd index j with a[j] even; swap")]
    for i in range(0, n, 2):
        if a[i] % 2 == 1:
            j = 1
            while j < n and a[j] % 2 == 1:
                j += 2
            a[i], a[j] = a[j], a[i]
            steps.append(_step(f"swap at i={i}, j={j}",
                               [_row("a", a, hi=i)],
                               [("i", i, "indigo"), ("j", j, "amber")],
                               eq=f"violating even slot {i} fixed via swap with {j}"))
        else:
            steps.append(_step(f"i={i} ok", [_row("a", a, hi=i)],
                            [("i", i, "indigo")], eq=""))
    assert a == [4, 5, 2, 7]
    return _pack("Sort Array by Parity II - pair-scan (inner loop)",
                 f"a = {a}", steps, str(a))


def _parity_twoptr():
    """Two independent step-by-2 cursors; each advances when satisfied, swap otherwise."""
    a = [3, 4, 1, 2]
    n = len(a)
    i, j = 0, 1
    steps = [_step("input", [_row("a", a)],
                  [("i", i, "indigo"), ("j", j, "amber")],
                  eq="i on even slots, j on odd slots; both advance while satisfied; swap on joint violation")]
    while i < n and j < n:
        while i < n and a[i] % 2 == 0:
            i += 2
        while j < n and a[j] % 2 == 1:
            j += 2
        if i < n and j < n:
            a[i], a[j] = a[j], a[i]
            steps.append(_step(f"swap i={i}, j={j}", [_row("a", a, hi=i)],
                                [("i", i, "indigo"), ("j", j, "amber")],
                                eq="both violated -> swap; each cursor is satisfied after"))
            i += 2
            j += 2
    assert a == [4, 3, 2, 1]
    return _pack("Sort Array by Parity II - two independent cursors (O(1) space)",
                 f"a = {[3, 4, 1, 2]}", steps, str(a))


# ----------------------------------------------------- LC 31 next permutation

_NP_A = [1, 3, 5, 4, 2]


def _nperm_brute():
    """Generate every permutation, sort them, find current, take next."""
    a = list(_NP_A)
    n = len(a)
    perms = sorted(itertools.permutations(a))
    k = perms.index(tuple(a))
    steps = [_step("input", [_row("a", a)],
                   eq=f"list every permutation of {a} in lex order; take the one AFTER the input")]
    for i in (max(0, k - 1), k, k + 1, min(len(perms) - 1, k + 2)):
        pass_no = f"perms[{i}]"
        steps.append(_step(pass_no if i != k else f"perms[{i}] = current",
                            [_row("a", a), _row("perm", list(perms[i]),
                                                 hi=None if i == k else None)],
                            eq="current at index " + str(k)))
    if len(steps) > 8:
        steps = [steps[0]] + steps[-7:]
    nxt = list(perms[(k + 1) % len(perms)])
    assert nxt == [1, 4, 2, 3, 5]
    return _pack("Next Permutation - enumerate all (small n only)",
                 f"a = {a}", steps, str(nxt))


def _nperm_sortsuf():
    """Find pivot, find successor, swap, then SORT the suffix. Correct but O(n log n)."""
    a = list(_NP_A)
    n = len(a)
    steps = [_step("input", [_row("a", a)],
                  eq="step 1: pivot = rightmost i with a[i] < a[i+1]")]
    i = n - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    steps.append(_step(f"pivot = {i}", [_row("a", a, hi=i)],
                        eq=f"a[{i}]={a[i]} < a[{i+1}]={a[i+1]}" if i >= 0 else "no pivot -> reverse all"))
    if i < 0:
        a.reverse()
        return _pack("Next Permutation - sort-suffix (wraparound)",
                     f"a = {list(_NP_A)}", steps, str(a))
    j = n - 1
    while a[j] <= a[i]:
        j -= 1
    a[i], a[j] = a[j], a[i]
    steps.append(_step(f"successor swap with j={j}", [_row("a", a, hi=i)],
                        eq=f"a[{j}]={a[j]} is rightmost > a[{i}] (before swap); swap; suffix is now a valid permutation"))
    a[i + 1:] = sorted(a[i + 1:])
    steps.append(_step("sort suffix", [_row("a", a, hi=i + 1)],
                        eq=f"sort a[{i+1}..n) ascending"))
    assert a == [1, 4, 2, 3, 5]
    return _pack("Next Permutation - pivot + successor + sort suffix",
                 f"a = {_NP_A}", steps, str(a))


def _nperm_optimal():
    """The 4-step: pivot, successor, swap, REVERSE suffix. O(n)."""
    a = list(_NP_A)
    n = len(a)
    steps = [_step("input", [_row("a", a)],
                  eq="step 1: pivot = rightmost i with a[i] < a[i+1] (suffix a[i+1..n) is non-increasing)")]
    i = n - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    steps.append(_step(f"pivot i={i}", [_row("a", a, hi=i)],
                        eq=f"a[{i}]={a[i]} < a[{i+1}]={a[i+1]}"))
    j = n - 1
    while a[j] <= a[i]:
        j -= 1
    steps.append(_step(f"successor j={j}", [_row("a", a, hi=j)],
                        eq=f"a[{j}]={a[j]} is rightmost with a[j] > a[i]"))
    a[i], a[j] = a[j], a[i]
    steps.append(_step("swap", [_row("a", a, hi=i)],
                    eq=f"swap a[{i}] <-> a[{j}]; suffix is STILL non-increasing"))
    lo, hi = i + 1, n - 1
    while lo < hi:
        a[lo], a[hi] = a[hi], a[lo]
        lo += 1
        hi -= 1
    steps.append(_step("reverse suffix", [_row("a", a, hi=i + 1)],
                        eq=f"reverse a[{i+1}..n) -> ascending; done"))
    assert a == [1, 4, 2, 3, 5]
    return _pack("Next Permutation - optimal 4-step (O(n))",
                 f"a = {_NP_A}", steps, str(a))


# --------------------------------------------- LC 556 next greater element III

_NG_N = 1244


def _nge_brute():
    """Enumerate digit permutations, find smallest > n."""
    n = _NG_N
    s = str(n)
    perms = sorted(set(int("".join(p)) for p in itertools.permutations(s)))
    nxt = None
    for x in perms:
        if x > n:
            nxt = x
            break
    steps = [_step("input", [_row("digits", list(s))],
                   eq=f"n = {n}; generate every permutation of its digits")]
    for k in (0, 1, 2):
        if k < len(perms) and perms[k] <= n:
            steps.append(_step(f"perm {perms[k]}", [_row("digits", list(s))],
                                eq=f"{perms[k]} <= {n}, skip"))
    steps.append(_step(f"first > {n}", [_row("digits", list(s))],
                        eq=f"{nxt} -> answer"))
    if len(steps) > 8:
        steps = [steps[0]] + steps[-7:]
    assert nxt == 1424
    return _pack("Next Greater Element III - enumerate permutations",
                 f"n = {n}", steps, str(nxt))


def _nge_sortsuf():
    """Apply the sort-suffix rung on the digit string. Same O(d log d)."""
    n = _NG_N
    d = list(str(n))
    steps = [_step("digits", [_row("d", d)],
                  eq="treat digits as an array of chars; apply the pivot/successor/sort-suffix form")]
    i = len(d) - 2
    while i >= 0 and d[i] >= d[i + 1]:
        i -= 1
    if i < 0:
        steps.append(_step("no pivot", [_row("d", d)], eq="digits are non-increasing -> next perm wraps to smallest -> -1"))
        return _pack("Next Greater Element III - sort-suffix on digits", f"n = {n}", steps, "-1")
    j = len(d) - 1
    while d[j] <= d[i]:
        j -= 1
    d[i], d[j] = d[j], d[i]
    steps.append(_step(f"swap i={i}, j={j}", [_row("d", d, hi=i)],
                        eq=f"pivot digit {d[j]!r} now at position {i}"))
    d[i + 1:] = sorted(d[i + 1:])
    steps.append(_step(f"sort suffix d[{i+1}:]", [_row("d", d, hi=i + 1)],
                        eq="sort ascending"))
    res = int("".join(d))
    limit = 2 ** 31 - 1
    ans = res if res <= limit else -1
    assert ans == 1424
    return _pack("Next Greater Element III - sort-suffix on digits",
                 f"n = {n}", steps, str(ans))


def _nge_optimal():
    """LC 31's 4-step on digits + overflow check."""
    n = _NG_N
    d = list(str(n))
    steps = [_step("digits", [_row("d", d)],
                  eq="pivot -> successor -> swap -> REVERSE suffix; then overflow check")]
    i = len(d) - 2
    while i >= 0 and d[i] >= d[i + 1]:
        i -= 1
    steps.append(_step(f"pivot i={i}", [_row("d", d, hi=i)],
                        eq=f"'{d[i]}' < '{d[i+1]}'"))
    j = len(d) - 1
    while d[j] <= d[i]:
        j -= 1
    steps.append(_step(f"successor j={j}", [_row("d", d, hi=j)], eq=f"'{d[j]}' > '{d[i]}'"))
    d[i], d[j] = d[j], d[i]
    steps.append(_step("swap", [_row("d", d, hi=i)], eq="swap in place"))
    lo, hi = i + 1, len(d) - 1
    while lo < hi:
        d[lo], d[hi] = d[hi], d[lo]
        lo += 1
        hi -= 1
    steps.append(_step("reverse suffix", [_row("d", d, hi=i + 1)], eq="reverse -> ascending (a non-decreasing suffix was already the reversed state)"))
    res = int("".join(d))
    limit = 2 ** 31 - 1
    ans = res if res <= limit else -1
    assert ans == 1424
    return _pack("Next Greater Element III - optimal 4-step on digits",
                 f"n = {n}", steps, str(ans))


# -------------------------------------------------- LC 324 wiggle sort II

def _wiggle_ok(a):
    return all((a[i] < a[i + 1]) if i % 2 == 0 else (a[i] > a[i + 1]) for i in range(len(a) - 1))


def _wiggle_dupfail():
    """Sort + front-interleave: the halves share the value 2, and the two copies end up adjacent."""
    a = [1, 2, 2, 3]
    s = sorted(a)
    n = len(s)
    half = (n + 1) // 2
    small, large = s[:half], s[half:]
    res = [0] * n
    for i in range(n):
        res[i] = small[i // 2] if i % 2 == 0 else large[i // 2]
    steps = [_step("sorted", [_row("s", s)], eq=f"half = (n+1)//2 = {half}")]
    steps.append(_step("halves", [_row("small", small), _row("large", large)],
                       eq=f"small = {small}, large = {large}: the value 2 is in both"))
    steps.append(_step("interleave from the front", [_row("res", res)],
                       eq="even slots take small[0], small[1]; odd slots take large[0], large[1]"))
    bad = next(i for i in range(n - 1)
               if (res[i] >= res[i + 1] if i % 2 == 0 else res[i] <= res[i + 1]))
    steps.append(_step(f"violation at slots {bad}, {bad + 1}", [_row("res", res, hi=bad)],
                       eq=f"need res[{bad}] > res[{bad + 1}], but both are {res[bad]}"))
    assert bad == 1 and not _wiggle_ok(res)
    return _pack("Wiggle Sort II - front interleave fails", f"nums = {a}", steps, str(res))


def _wiggle_sortrev():
    """Sort, then deal the small half into even slots and the large half into odd slots, both reversed."""
    a = [1, 3, 2, 2, 3, 1]
    s = sorted(a)
    n = len(s)
    half = (n + 1) // 2
    small, large = s[:half][::-1], s[half:][::-1]
    res = ["."] * n
    steps = [_step("sorted", [_row("s", s)],
                   eq=f"small = s[:{half}] reversed = {small}; large = s[{half}:] reversed = {large}")]
    for i, v in enumerate(small):
        res[2 * i] = v
    steps.append(_step("even slots", [_row("res", res, hi=0)],
                       eq=f"res[0], res[2], ... = {small}: the largest small value goes first"))
    for i, v in enumerate(large):
        res[2 * i + 1] = v
    steps.append(_step("odd slots", [_row("res", res, hi=1)],
                       eq=f"res[1], res[3], ... = {large}: the copies of 2 are now far apart"))
    assert _wiggle_ok(res) and res == [2, 3, 1, 3, 1, 2]
    return _pack("Wiggle Sort II - sort, reversed halves", f"nums = {a}", steps, str(res))


def _wiggle_virtual():
    """Three-way partition run through the virtual index A(i) = nums[(1 + 2i) % (n | 1)]."""
    a = [1, 3, 2, 2, 3, 1]
    n = len(a)
    med = sorted(a)[(n - 1) // 2]
    arr = list(a)
    m = n | 1

    def vi(i):
        return (1 + 2 * i) % m

    lo, mid, hi = 0, 0, n - 1
    steps = [_step("input", [_row("nums", arr)],
                   [("m", med, "amber"), ("i", vi(lo), "green"), ("j", vi(mid), "indigo"),
                    ("k", vi(hi), "violet")],
                   eq=f"median m = {med}; virtual slot t lives at real index (1 + 2t) % {m}")]
    while mid <= hi:
        x = arr[vi(mid)]
        if x > med:
            arr[vi(lo)], arr[vi(mid)] = arr[vi(mid)], arr[vi(lo)]
            what = f"{x} > m: swap real {vi(lo)}, {vi(mid)}; i++, j++"
            lo += 1
            mid += 1
        elif x < med:
            arr[vi(mid)], arr[vi(hi)] = arr[vi(hi)], arr[vi(mid)]
            what = f"{x} < m: swap real {vi(mid)}, {vi(hi)}; k--"
            hi -= 1
        else:
            what = f"{x} == m: j++"
            mid += 1
        steps.append(_step(what, [_row("nums", arr)],
                           [("i", vi(lo) if lo < n else "-", "green"),
                            ("j", vi(mid) if mid < n else "-", "indigo"),
                            ("k", vi(hi) if hi >= 0 else "-", "violet")], eq=what))
    assert _wiggle_ok(arr) and arr == [2, 3, 1, 3, 1, 2]
    return _pack("Wiggle Sort II - median + virtual-index partition, in place",
                 f"nums = {a}", steps, str(arr))


# ---------------------------------------------- CF 1863B Split Sort (Pattern 1)

def _ss_apply(p, x):
    """One Split-Sort operation: stable write of <x, then >=x."""
    return [v for v in p if v < x] + [v for v in p if v >= x]


def _ss_descents(p):
    """Values v with pos(v+1) before pos(v) - the pairs one op must fix."""
    pos = {v: i for i, v in enumerate(p)}
    return [v for v in range(1, len(p)) if pos[v + 1] < pos[v]]


def _splitsort_bfs():
    """Threshold BFS over permutations - exact for n = 6, the definition-level rung."""
    start = [6, 4, 3, 5, 2, 1]
    target = list(range(1, 7))
    steps = [_step("input", [_row("p", start)],
                   [("ops", 0, "indigo")],
                   eq="level 0: choose x in 2..n; p -> [< x in order] + [>= x in order]")]
    frontier = {tuple(start)}
    seen = {tuple(start)}
    ops = 0
    while tuple(sorted(frontier)) != tuple(target) and len(steps) < 6:
        ops += 1
        nxt = set()
        for st in frontier:
            for x in range(2, 7):
                cand = tuple(_ss_apply(list(st), x))
                if cand == tuple(target):
                    steps.append(_step(f"level {ops}: x = {x} hits sorted",
                                       [_row("p", list(cand), hi=0)],
                                       [("ops", ops, "green")],
                                       eq=f"{list(st)} -> x={x} -> {list(cand)} = target"))
                    steps.append(_step("answer",
                                       [_row("p", list(cand), hi=i) for i in range(1)],
                                       [("ops", ops, "green")],
                                       eq=f"minimum operations = {ops}"))
                    return _pack("Split Sort - threshold BFS (oracle)",
                                 f"p = {start}", steps, str(ops))
                if cand not in seen:
                    seen.add(cand)
                    nxt.add(cand)
        sample = sorted(nxt)[0] if nxt else None
        if sample is not None:
            steps.append(_step(f"level {ops} (first of {len(nxt)} states)",
                               [_row("p", list(sample))],
                               [("ops", ops, "indigo")],
                               eq=f"example reachable state: {list(sample)}"))
        frontier = nxt
    raise AssertionError("BFS did not reach target")


def _splitsort_descents():
    """Count descents of the inverse permutation - the answer formula rung."""
    p = [6, 4, 3, 5, 2, 1]
    n = len(p)
    pos = {v: i for i, v in enumerate(p)}
    inv = [pos[v] for v in range(1, n + 1)]
    steps = [_step("input", [_row("p", p)],
                   eq="pos[v] = index of value v; a pair with pos(v+1) < pos(v) needs one op")]
    steps.append(_step("inverse pos", [_row("pos", inv)],
                       eq=f"pos = {inv} for values 1..{n}"))
    desc = _ss_descents(p)
    for v in range(1, n):
        bad = pos[v + 1] < pos[v]
        steps.append(_step(f"pair ({v}, {v + 1})", [_row("pos", inv, hi=v - 1)],
                           [("pos[v]", pos[v], "indigo"), ("pos[v+1]", pos[v + 1],
                                                           "amber" if bad else "green")],
                           eq=(f"{pos[v + 1]} < {pos[v]}: descent, needs x = {v + 1}" if bad
                               else f"{pos[v + 1]} > {pos[v]}: already in order")))
    big = [10, 19, 7, 1, 17, 11, 8, 5, 12, 9, 4, 18, 14, 2, 6, 15, 3, 16, 13]
    assert len(_ss_descents(big)) == 7
    assert desc == [1, 2, 3, 5]
    assert _ss_descents([1]) == []
    assert _ss_descents([2, 1]) == [1]
    return _pack("Split Sort - descents of the inverse",
                 f"p = {p}", steps, str(len(desc)))


def _splitsort_execute():
    """Actually run the ops a greedy picks from the descents, to sorted."""
    p = [6, 4, 3, 5, 2, 1]
    target = list(range(1, 7))
    ops = 0
    steps = [_step("input", [_row("p", p)],
                   eq="while not sorted: pick x = v+1 for some remaining descent v")]
    while p != target:
        desc = _ss_descents(p)
        assert desc, "sortedness always reachable via descents"
        x = desc[0] + 1
        p = _ss_apply(p, x)
        ops += 1
        steps.append(_step(f"op {ops}: x = {x}",
                           [_row("p", p, hi=p.index(x))],
                           [("x", x, "green"), ("desc left", len(desc) - 1, "amber")],
                           eq=f"stable split at {x}: descent {desc[0]} repaired"))
        if len(steps) >= 7:
            break
    steps.append(_step("answer", [_row("p", p, hi=i) for i in [0]],
                       [("ops", ops, "green")], eq=f"{ops} operations"))
    return _pack("Split Sort - executing the descent repair",
                 "p = [6, 4, 3, 5, 2, 1]", steps, str(ops))


# --------------------------------- CF 2037C Superultra (Pattern 4, wiggle-adjacent)

def _su_ok(p):
    """All adjacent sums composite? (4 is the smallest reachable sum.)"""
    def composite(x):
        return x > 4 and any(x % d == 0 for d in range(2, int(x ** 0.5) + 1)) or x == 4
    return all(composite(p[i] + p[i + 1]) for i in range(len(p) - 1))


def _superultra_brute():
    """Lexicographic permutation scan; n = 3 has no answer -> -1 (official sample)."""
    import itertools
    n = 3
    steps = [_step("input", [_row("values", [1, 2, 3])],
                   eq="scan all 6 permutations; need every adjacent sum composite")]
    checked = 0
    for perm in itertools.permutations(range(1, n + 1)):
        checked += 1
        s0, s1 = perm[0] + perm[1], perm[1] + perm[2]
        steps.append(_step(f"perm {checked}",
                           [_row("p", list(perm))],
                           [("sums", f"{s0},{s1}", "amber")],
                           eq=f"{s0} and {s1}: {'ok' if _su_ok(list(perm)) else 'some sum is prime'}"))
        if _su_ok(list(perm)):
            return _pack("Superultra - lexicographic brute", f"n = {n}", steps, str(list(perm)))
        if len(steps) >= 7:
            break
    steps.append(_step("exhausted", [_row("values", [1, 2, 3])],
                       [("tried", checked, "amber")],
                       eq=f"first {len(steps) - 1} of 6 shown; all 6 fail -> answer -1"))
    return _pack("Superultra - lexicographic brute (n=3 fails)", f"n = {n}", steps, "-1")


def _superultra_evens():
    """Evens ascending, then 1, then the other odds: the seam E + 1 decides everything."""
    n = 6
    evens = list(range(2, n + 1, 2))
    odds = list(range(1, n + 1, 2))
    p = evens + [1] + [o for o in odds if o != 1]
    pad = len(p) - len(evens)
    steps = [_step("evens block", [_row("p", evens + ["."] * pad)],
                   eq="2, 4, ..., E: every internal sum is even and >= 6 -> composite")]
    steps.append(_step("append odd head 1", [_row("p", evens + [1] + ["."] * (pad - 1))],
                       [("seam", evens[-1] + 1, "amber")],
                       eq=f"seam sum = last even {evens[-1]} + 1 = {evens[-1] + 1}"))
    good = _su_ok(p)
    steps.append(_step("seam check", [_row("p", p, hi=len(evens))],
                       [("seam", evens[-1] + 1, "green" if good else "red")],
                       eq=f"{evens[-1] + 1} is {'composite' if good else 'prime -> the construction FAILS'}"))
    steps.append(_step("odds tail", [_row("p", p)],
                       eq="odd + odd sums are even and >= 4 -> composite; only the seam can break"))
    assert not good
    return _pack("Superultra - evens then odds, head 1", f"n = {n}", steps,
                 "seam 7 is prime: invalid")


def _superultra_residue():
    """Choose the odd head o with (E + o) % 3 == 0 - works for every n >= 5."""
    n = 6
    evens = list(range(2, n + 1, 2))
    last = evens[-1]
    odds = list(range(1, n + 1, 2))
    r = (-last) % 3
    o1 = next(o for o in odds if o % 3 == r)
    p = evens + [o1] + [o for o in odds if o != o1]
    steps = [_step("target residue", [_row("evens", evens)],
                   [("r", r, "indigo")],
                   eq=f"want head o with ({last} + o) % 3 == 0, so o % 3 = {r}"),
             _step("pick head", [_row("odds", odds, hi=odds.index(o1))],
                   [("head", o1, "green")],
                   eq=f"the odd in 1, 3, 5 with o % 3 == {r} is {o1}")]
    steps.append(_step("assembled", [_row("p", p, hi=len(evens))],
                       [("seam", last + o1, "green")],
                       eq=f"{last} + {o1} = {last + o1}: a multiple of 3 above 3 -> composite"))
    sums = [p[i] + p[i + 1] for i in range(n - 1)]
    steps.append(_step("verify", [_row("p", p), _row("sums", sums)],
                       eq=f"adjacent sums {sums}: all composite"))
    assert p == [2, 4, 6, 3, 1, 5]
    assert _su_ok(p)
    return _pack("Superultra - residue-chosen odd head", f"n = {n}", steps, str(p))


# ------------------------------------------ CF 1579B Shifting Sort (Pattern 5)

def _shiftsort_ops():
    """Selection sort where each placement is one left rotation of a[i..j] by j - i."""
    a = [2, 5, 1, 4, 3]
    n = len(a)
    cur = list(a)
    ops = []
    steps = [_step("input", [_row("a", cur)], [("ops", 0, "indigo")],
                   eq="for i = 0..n-1: bring the minimum of a[i..] to i with ONE rotation")]
    for i in range(n):
        j = min(range(i, n), key=lambda t: (cur[t], t))
        if j == i:
            steps.append(_step(f"i = {i}: already the minimum", [_row("a", cur, hi=i)],
                               [("ops", len(ops), "indigo")], eq=f"a[{i}] = {cur[i]}, no operation"))
            continue
        d = j - i
        cur = cur[:i] + cur[j:j + 1] + cur[i:j] + cur[j + 1:]
        ops.append((i + 1, j + 1, d))
        steps.append(_step(f"i = {i}: rotate [{i + 1}, {j + 1}] left by {d}", [_row("a", cur, hi=i)],
                           [("ops", len(ops), "green")],
                           eq=f"op ({i + 1}, {j + 1}, {d}): a[{j}] = {cur[i]} jumps to index {i}"))
    assert cur == sorted(a) and len(ops) == 3
    return _pack("Shifting Sort - one rotation per placement", f"a = {a}", steps, str(ops))


# ------------------------------------- CF 1969B Shifts and Sorting (Pattern 5)

def _shifts_dijkstra():
    """Exact shortest-cost search over substring rotations for s = '101'."""
    import heapq
    s = "101"
    target = "".join(sorted(s))
    dist = {s: 0}
    pq = [(0, s)]
    steps = [_step("input", [_row("s", list(s))],
                   [("cost", 0, "indigo")],
                   eq="edges: rotate any substring s[l..r] right once, cost r-l+1")]
    settled = 0
    while pq:
        d, u = heapq.heappop(pq)
        if dist.get(u, d) != d:
            continue
        settled += 1
        if u == target:
            steps.append(_step(f"settled #{settled}: target",
                               [_row("s", list(u))],
                               [("cost", d, "green")],
                               eq=f"011 reached with total cost {d}"))
            assert d == 2
            return _pack("Shifts and Sorting - Dijkstra over rotations",
                         f"s = {s}", steps, str(d))
        moves = []
        for l in range(len(u)):
            for r in range(l + 1, len(u)):
                v = u[:l] + u[r] + u[l:r] + u[r + 1:]
                nd = d + (r - l + 1)
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
                    moves.append((f"[{l},{r}]", v, nd))
        if len(steps) < 6:
            mv = moves[0] if moves else ("-", u, d)
            steps.append(_step(f"settled #{settled}: {u}",
                               [_row("s", list(u))],
                               [("best", f"{mv[0]} -> {mv[1]} cost {mv[2]}", "amber")],
                               eq=f"relax {len(moves)} rotations; e.g. {mv[0]}: {u} -> {mv[1]} (+{mv[0] and mv[2] - d})"))
    raise AssertionError("target unreachable")


def _shifts_blocks():
    """The block strategy from the note: rotate each 1-run plus the next 0."""
    s = "11000"
    cur = list(s)
    total = 0
    steps = [_step("input", [_row("s", cur)],
                   [("cost", 0, "indigo")],
                   eq="while a '1' precedes a '0': rotate the 1-run plus that 0 right once")]
    guard = 0
    while any(cur[i] == "1" and cur[j] == "0" for i in range(len(cur)) for j in range(i + 1, len(cur))):
        guard += 1
        assert guard < 10
        i1 = cur.index("1")
        j = i1
        while j < len(cur) and cur[j] == "1":
            j += 1
        length = j - i1 + 1
        cur = cur[:i1] + [cur[j]] + cur[i1:j] + cur[j + 1:]
        total += length
        steps.append(_step(f"rotate [{i1},{j}] len {length}",
                           [_row("s", cur, hi=i1)],
                           [("cost", total, "amber")],
                           eq=f"run of {j - i1} ones + one zero, cyclic right shift"))
        if len(steps) >= 7:
            break
    steps.append(_step("answer", [_row("s", cur)],
                       [("cost", total, "green")], eq=f"sorted 00011 with total cost {total}"))
    assert "".join(cur) == "00011" and total == 9
    return _pack("Shifts and Sorting - block rotation strategy", f"s = {s}", steps, str(total))


def _shifts_scan():
    """One pass: every 0 that has k>0 ones to its left pays k+1 (no reset).

    This is the verified closed form: '11000' pays 3 + 3 + 3 = 9,
    '101011' pays 2 + 3 = 5, '01101001' pays 3 + 4 + 4 = 11."""
    s = "11000"
    ones = 0
    cost = 0
    trace = []
    steps = [_step("input", [_row("s", list(s))],
                   [("ones", 0, "indigo"), ("cost", 0, "green")],
                   eq="scan left to right; count ones; each 0 with ones>0 pays ones+1")]
    for i, ch in enumerate(s):
        if ch == "1":
            ones += 1
            eq = f"see 1: ones = {ones}"
        elif ones:
            add = ones + 1
            cost += add
            eq = f"see 0 with {ones} ones to its left: pay {ones}+1 = {add}, cost = {cost}"
        else:
            eq = "see 0 with no ones to its left: free"
        trace.append(_step(f"i = {i}", [_row("s", list(s), hi=i)],
                           [("ones", ones, "indigo"), ("cost", cost, "green")], eq=eq))
    steps += trace[-6:]
    assert cost == 9
    return _pack("Shifts and Sorting - closed-form scan", f"s = {s}", steps, str(cost))


# ------------------------------------------ CF 1365C Rotation Matching (Pattern 5)

def _rotation_brute():
    """Try every relative rotation of b against a - O(n^2) definition rung."""
    from collections import deque
    a = [5, 4, 3, 2, 1]
    b = [1, 2, 3, 4, 5]
    n = len(a)
    dq = deque(b)
    best = 0
    best_k = 0
    steps = [_step("input", [_row("a", a), _row("b", list(dq))],
                   eq="rotate b right by k = 0..n-1; count fixed points for each k")]
    for k in range(n):
        m = sum(1 for i in range(n) if a[i] == dq[i])
        if m > best:
            best, best_k = m, k
        steps.append(_step(f"k = {k}", [_row("a", a), _row("b", list(dq), hi=n // 2)],
                           [("match", m, "green" if m == best else "amber")],
                           eq=f"{m} matches at shift {k}"))
        dq.rotate(1)
        if len(steps) >= 7:
            break
    steps.append(_step("answer", [_row("a", a), _row("b", b)],
                       [("best", best, "green")],
                       eq=f"max over all k: {best} (at k = {best_k})"))
    assert best == 1
    return _pack("Rotation Matching - try every shift (oracle)",
                 f"a = {a}, b = {b}", steps, str(best))


def _rotation_offsets():
    """Each value votes for the rotation (pos_a - pos_b) mod n that frees it."""
    a = [1, 3, 2, 4]
    b = [4, 2, 3, 1]
    n = len(a)
    pa = {v: i for i, v in enumerate(a)}
    pb = {v: i for i, v in enumerate(b)}
    cnt = {}
    steps = [_step("input", [_row("a", a), _row("b", b)],
                   eq="value v matches under rotation k iff k = (pos_a(v) - pos_b(v)) mod n")]
    for v in range(1, n + 1):
        k = (pa[v] - pb[v]) % n
        cnt[k] = cnt.get(k, 0) + 1
        steps.append(_step(f"value {v}",
                           [_row("a", a, hi=pa[v]), _row("b", b, hi=pb[v])],
                           [("k", k, "indigo"), ("cnt", cnt[k], "green")],
                           eq=f"pos_a={pa[v]}, pos_b={pb[v]}, vote k = {k}"))
    best = max(cnt.values())
    steps.append(_step("histogram", [_row("a", a), _row("b", b)],
                       [(f"cnt[k={kk}]", vv, "green" if vv == best else "amber")
                        for kk, vv in sorted(cnt.items())],
                       eq=f"answer = max count = {best}"))
    assert best == 2
    return _pack("Rotation Matching - offset histogram", f"a = {a}, b = {b}", steps, str(best))


def _rotation_realize():
    """Show the vote is realized: deque-rotate b by the winning k and count."""
    from collections import deque
    a = [1, 2, 3, 4, 5]
    b = [2, 3, 4, 5, 1]
    n = len(a)
    pa = {v: i for i, v in enumerate(a)}
    pb = {v: i for i, v in enumerate(b)}
    cnt = {}
    for v in range(1, n + 1):
        k = (pa[v] - pb[v]) % n
        cnt[k] = cnt.get(k, 0) + 1
    k = max(cnt, key=cnt.get)
    dq = deque(b)
    dq.rotate(k)
    m = sum(1 for i in range(n) if a[i] == dq[i])
    steps = [_step("votes", [_row("a", a), _row("b", b)],
                   [("k*", k, "green"), ("votes", cnt[k], "indigo")],
                   eq=f"every value votes k = 1; winning rotation k* = {k}")]
    steps.append(_step(f"rotate b right by {k}", [_row("a", a), _row("b", list(dq), hi=0)],
                       [("match", m, "green")],
                       eq=f"b becomes {list(dq)} - all {m} positions match"))
    assert m == 5 == cnt[k]
    return _pack("Rotation Matching - realizing the winning shift",
                 f"a = {a}, b = {b}", steps, str(m))


# ------------------------------------------------------------------ registry

REGISTRY = {
    ("pivot", "brute"): _pivot_brute(),
    ("pivot", "threelist"): _pivot_threelist(),
    ("pivot", "twoend"): _pivot_twoend(),
    ("pivot", "dnf"): _pivot_dnf(),
    ("signs", "brute"): _signs_brute(),
    ("signs", "twoptr"): _signs_twoptr(),
    ("signs", "writecursors"): _signs_writecursors(),
    ("signs", "inplace"): _signs_inplace(),
    ("parity", "brute"): _parity_brute(),
    ("parity", "pairscan"): _parity_pairscan(),
    ("parity", "twoptr"): _parity_twoptr(),
    ("nperm", "brute"): _nperm_brute(),
    ("nperm", "sortsuf"): _nperm_sortsuf(),
    ("nperm", "optimal"): _nperm_optimal(),
    ("nge3", "brute"): _nge_brute(),
    ("nge3", "sortsuf"): _nge_sortsuf(),
    ("nge3", "optimal"): _nge_optimal(),
    ("wiggle", "dupfail"): _wiggle_dupfail(),
    ("wiggle", "sortrev"): _wiggle_sortrev(),
    ("wiggle", "virtual"): _wiggle_virtual(),
    ("splitsort", "bfs"): _splitsort_bfs(),
    ("splitsort", "descents"): _splitsort_descents(),
    ("splitsort", "execute"): _splitsort_execute(),
    ("superultra", "brute"): _superultra_brute(),
    ("superultra", "evens"): _superultra_evens(),
    ("superultra", "residue"): _superultra_residue(),
    ("shiftsort", "ops"): _shiftsort_ops(),
    ("shifts", "dijkstra"): _shifts_dijkstra(),
    ("shifts", "blocks"): _shifts_blocks(),
    ("shifts", "scan"): _shifts_scan(),
    ("rotation", "brute"): _rotation_brute(),
    ("rotation", "offsets"): _rotation_offsets(),
    ("rotation", "realize"): _rotation_realize(),
}


def get(problem, form):
    return REGISTRY[(problem, form)]
