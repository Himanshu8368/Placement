"""Step frames for the sliding-window chapter.

Drawing lives in ``dsa_viz.story``. ``get`` returns data only.
Each story walks the chapter's own sample, at most eight frames, and the
numbers come from running the algorithm - never from a hand-typed table.
A window frame shows the array row with the right edge highlighted plus the
rolling chips (left edge, window aggregate, best), so the reader watches the
window grow and shrink in step with the arithmetic.
"""
from __future__ import annotations

import bisect
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import chip_step, row_step, show


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    if not 1 <= len(steps) <= 8:
        raise AssertionError(f"{title}: {len(steps)} frames")
    return (title, inp, steps, text)


def _step(title, rows, chips=None, eq=""):
    """One frame: an array row (or two) plus the moving window as chips."""
    frame = row_step(title, rows, eq=eq)
    if chips:
        frame["chips"] = chip_step(title, chips, eq="")["chips"]
    return frame


def _avg_fixed():
    nums, k = [1, 12, -5, -6, 50, 3], 4
    cur = sum(nums[:k])
    best = cur
    steps = [_step(f"window [0..{k - 1}]", [("nums", nums, k - 1)],
                   [("l", 0, "indigo"), ("r", k - 1, "violet"), ("sum", cur, "amber"),
                    ("best", best, "green")],
                   eq=f"first window sum = {cur}")]
    for l in range(1, len(nums) - k + 1):
        r = l + k - 1
        cur = cur - nums[l - 1] + nums[r]
        best = max(best, cur)
        steps.append(_step(f"window [{l}..{r}]", [("nums", nums, r)],
                           [("l", l, "indigo"), ("r", r, "violet"), ("sum", cur, "amber"),
                            ("best", best, "green")],
                           eq=f"drop {nums[l - 1]}, add {nums[r]} -> sum {cur}"))
    return _pack("Fixed-size window - slide by one, drop the leaver, add the enterer",
                 "nums = [1, 12, -5, -6, 50, 3], k = 4", steps, best / k)


def _avg_prefix():
    nums, k = [1, 12, -5, -6, 50, 3], 4
    P = [0]
    for x in nums:
        P.append(P[-1] + x)
    assert P == [0, 1, 13, 8, 2, 52, 55]
    steps = []
    best = -math.inf
    for l in range(len(nums) - k + 1):
        s = P[l + k] - P[l]
        best = max(best, s)
        steps.append(_step(f"query [{l}..{l + k - 1}]", [("P", P, l + k)],
                           [("P[l+k]", P[l + k], "violet"), ("P[l]", P[l], "indigo"),
                            ("sum", s, "amber"), ("best", best, "green")],
                           eq=f"P[{l + k}] - P[{l}] = {P[l + k]} - {P[l]} = {s}; avg {s / k}"))
    return _pack("Prefix sums - every window is one subtraction on a stored table",
                 "nums = [1, 12, -5, -6, 50, 3], k = 4; P = [0, 1, 13, 8, 2, 52, 55]",
                 steps, best / k)


ONES_NUMS, ONES_K = [1, 0, 0, 1, 0, 1, 1], 2      # the board sample of LC 1004


def _ones_grow():
    nums, k = ONES_NUMS, ONES_K
    zeros = l = best = 0
    steps = []
    for r, x in enumerate(nums):
        zeros += x == 0
        moved = 0
        while zeros > k:
            zeros -= nums[l] == 0
            l += 1
            moved += 1
        best = max(best, r - l + 1)
        tag = f" (shrink {moved})" if moved else ""
        eq = (f"over budget: l crept {moved} to {l}; " if moved else "") + \
            f"[{l}..{r}] has {zeros} zeros, length {r - l + 1}"
        steps.append(_step(f"r = {r}{tag}", [("nums", nums, r)],
                           [("l", l, "indigo"), ("r", r, "violet"), ("zeros", zeros, "amber"),
                            ("best", best, "green")],
                           eq=eq))
    assert best == 5
    return _pack("Longest valid window - shrink only when the budget breaks",
                 f"nums = {nums}, k = {k} (flip budget)", steps, best)


def _ones_bs():
    nums, k = ONES_NUMS, ONES_K
    Z = [0]
    for x in nums:
        Z.append(Z[-1] + (x == 0))
    n = len(nums)

    def fits(L):
        """Is some length-L window within the zero budget? Returns (ok, start)."""
        for i in range(n - L + 1):
            if Z[i + L] - Z[i] <= k:
                return True, i
        return False, 0

    lo, hi, ans, steps = 1, n, 0, []
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, i = fits(mid)
        z = Z[i + mid] - Z[i]
        if ok:
            ans, lo = mid, mid + 1
            note = f"Z[{i + mid}] - Z[{i}] = {Z[i + mid]} - {Z[i]} = {z} <= {k}: len {mid} fits, search longer"
        else:
            z = min(Z[i + mid] - Z[i] for i in range(n - mid + 1))
            hi = mid - 1
            note = f"every length-{mid} window holds >= {z} > {k} zeros: too long, search shorter"
        steps.append(_step(f"check len {mid}", [("Z", Z, i + mid if ok else None)],
                           [("len", mid, "indigo"), ("zeros", z, "amber"),
                            ("fits", "yes" if ok else "no", "green" if ok else "red"),
                            ("answer", ans, "green")],
                           eq=note))
    steps.append(_step("binary search ends", [("Z", Z, None)],
                       [("lo", lo, "muted"), ("hi", hi, "muted"), ("answer", ans, "green")],
                       eq="lo crossed hi: the longest feasible length is settled"))
    assert ans == 5 and lo == hi + 1
    return _pack("Binary search on the length - 'is a window this long feasible?'",
                 f"nums = {nums}, k = {k}; zero-prefix Z = {Z}",
                 steps, ans)


def _ones_deque():
    nums, k = ONES_NUMS, ONES_K
    dq = []
    l = best = 0
    steps = []
    for r, x in enumerate(nums):
        jumped = False
        if x == 0:
            dq.append(r)
            if len(dq) > k:
                l = dq.pop(0) + 1
                jumped = True
        best = max(best, r - l + 1)
        assert all(nums[z] == 0 and l <= z <= r for z in dq)
        steps.append(_step(f"r = {r}" + (" (jump l)" if jumped else ""), [("nums", nums, r)],
                           [("l", l, "indigo"), ("r", r, "violet"),
                            ("zeros in window", list(dq), "amber"), ("best", best, "green")],
                           eq=(f"third zero arrives: l jumps past the oldest zero to {l}; "
                               f"window [{l}..{r}], length {r - l + 1}") if jumped
                           else f"window [{l}..{r}] holds {len(dq)} zeros <= {k}; length {r - l + 1}"))
    assert best == 5
    return _pack("Zero-index queue - the left bound jumps instead of creeping",
                 f"nums = {nums}, k = {k} (flip budget)", steps, best)


def _books_grow():
    a, t = [3, 1, 2, 1], 5
    l = s = best = 0
    steps = []
    for r, x in enumerate(a):
        s += x
        while s > t:
            s -= a[l]
            l += 1
        best = max(best, r - l + 1)
        steps.append(_step(f"r = {r}", [("a", a, r)],
                           [("l", l, "indigo"), ("r", r, "violet"), ("time", s, "amber"),
                            ("best", best, "green")],
                           eq=f"window [{l}..{r}] time {s} <= {t}; {r - l + 1} books"))
    return _pack("Books - longest run within a time budget, shrink while over",
                 "a = [3, 1, 2, 1], t = 5", steps, best)


def _books_bs():
    a, T = [3, 1, 2, 1], 5
    P = [0]
    for x in a:
        P.append(P[-1] + x)
    n = len(a)

    def fits(L):
        for i in range(n - L + 1):
            if P[i + L] - P[i] <= T:
                return True, i
        return False, 0

    lo, hi, ans, steps = 1, n, 0, []
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, i = fits(mid)
        if ok:
            s = P[i + mid] - P[i]
            ans, lo = mid, mid + 1
            eq = f"P[{i + mid}] - P[{i}] = {P[i + mid]} - {P[i]} = {s} <= {T}: {mid} books fit, try more"
        else:
            s = min(P[i + mid] - P[i] for i in range(n - mid + 1))
            hi = mid - 1
            eq = f"cheapest {mid}-book stretch costs {s} > {T}: try fewer"
        steps.append(_step(f"probe len {mid}", [("P", P, i + mid if ok else None)],
                           [("len", mid, "indigo"), ("sum", s, "amber"),
                            ("fits", "yes" if ok else "no", "green" if ok else "red"),
                            ("answer", ans, "green")],
                           eq=eq))
    assert ans == 3
    return _pack("Books by bisecting the count - feasibility from a prefix-sum table",
                 "t = [3, 1, 2, 1], T = 5; prefix P = [0, 3, 4, 6, 7]", steps, ans)


def _mindist_shrink():
    nums, target = [2, 3, 1, 2, 4, 3], 7
    l = s = 0
    best = math.inf
    best_l = 0
    steps = []
    for r, x in enumerate(nums):
        s += x
        logged = []
        while s >= target:
            logged.append(r - l + 1)
            if r - l + 1 < best:
                best, best_l = r - l + 1, l
            s -= nums[l]
            l += 1
        shown = "none" if math.isinf(best) else best
        if logged:
            eq = (f"valid -> logged length(s) {', '.join(map(str, logged))} while shrinking; "
                  f"now [{l}..{r}] sums to {s} < {target}")
        else:
            eq = f"[{l}..{r}] sums to {s} < {target}: not valid yet, grow"
        steps.append(_step(f"r = {r}", [("nums", nums, r)],
                           [("l", l, "indigo"), ("r", r, "violet"), ("sum", s, "amber"),
                            ("best", shown, "green")],
                           eq=eq))
    return _pack("Shortest valid window - shrink while still valid, record inside the while",
                 "nums = [2, 3, 1, 2, 4, 3], target = 7", steps, best)


def _mindist_bs():
    nums, target = [2, 3, 1, 2, 4, 3], 7
    P = [0]
    for x in nums:
        P.append(P[-1] + x)
    n = len(nums)

    def fits(L):
        """Some length-L window with sum >= target? Positivity makes this monotone down in L."""
        for i in range(n - L + 1):
            if P[i + L] - P[i] >= target:
                return True, i
        return False, 0

    lo, hi, ans, steps = 1, n, 0, []
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, i = fits(mid)
        if ok:
            s = P[i + mid] - P[i]
            ans, hi = mid, mid - 1            # length mid works: try SHORTER
            eq = f"P[{i + mid}] - P[{i}] = {P[i + mid]} - {P[i]} = {s} >= {target}: len {mid} works, hunt shorter"
        else:
            s = max(P[i + mid] - P[i] for i in range(n - mid + 1))
            lo = mid + 1                       # no window this short reaches: go longer
            eq = f"richest length-{mid} window holds only {s} < {target}: hunt longer"
        steps.append(_step(f"probe len {mid}", [("P", P, i + mid if ok else None)],
                           [("len", mid, "indigo"), ("sum", s, "amber"),
                            ("hits", "yes" if ok else "no", "green" if ok else "red"),
                            ("answer", ans, "green")],
                           eq=eq))
    assert ans == 2
    return _pack("Minimum Size - bisect the length, probe the prefix table",
                 "nums = [2, 3, 1, 2, 4, 3], target = 7; P = [0, 2, 5, 6, 8, 12, 15]", steps, ans)


def _product_count():
    nums, k = [10, 5, 2, 6], 100
    l = 0
    prod = 1
    cnt = 0
    steps = []
    for r, x in enumerate(nums):
        prod *= x
        while prod >= k and l <= r:
            prod //= nums[l]
            l += 1
        cnt += r - l + 1
        steps.append(_step(f"r = {r}", [("nums", nums, r)],
                           [("l", l, "indigo"), ("r", r, "violet"), ("prod", prod, "amber"),
                            ("count", cnt, "green")],
                           eq=f"product [{l}..{r}] = {prod} < {k}: {r - l + 1} new subarrays ending here"))
    return _pack("Subarray Product Less Than K - count = sum of window lengths",
                 "nums = [10, 5, 2, 6], k = 100", steps, cnt)


def _product_perleft():
    nums, k = [10, 5, 2, 6], 100
    n = len(nums)
    steps, total = [], 0
    for l in range(n):
        p = 1
        r = l - 1
        while r + 1 < n and p * nums[r + 1] < k:
            p *= nums[r + 1]
            r += 1
        total += r - l + 1
        steps.append(_step(f"left {l}: run right from there", [("nums", nums, r)],
                           [("l", l, "indigo"), ("stop at", r, "violet"), ("prod", p, "amber"),
                            ("count", total, "green")],
                           eq=(f"product of [{l}..{r}] = {p} < {k}; " + (f"multiplying in {nums[r + 1]} would reach {k}" if r + 1 < n else "the run reaches the end")
                              + f": +{r - l + 1}")))
    assert total == 8
    return _pack("Product < K, one fixed left at a time - the running product never rewinds",
                 "nums = [10, 5, 2, 6], k = 100", steps, total)


DISTINCT_NUMS, DISTINCT_K = [1, 5, 4, 2, 9, 9, 9], 3   # LC 2461 example 1


def _distinct_fixed():
    nums, k = DISTINCT_NUMS, DISTINCT_K
    win = Counter(nums[:k])
    cur = sum(nums[:k])
    dup = sum(c - 1 for c in win.values() if c > 1)
    best = cur if dup == 0 else None
    steps = [_step(f"window [0..{k - 1}]", [("nums", nums, k - 1)],
                   [("l", 0, "indigo"), ("r", k - 1, "violet"), ("sum", cur, "amber"),
                    ("dups", dup, "red"), ("best", best if best is not None else "-", "green")],
                   eq=f"first window sum {cur}, {dup} duplicate(s)")]
    for l in range(1, len(nums) - k + 1):
        r = l + k - 1
        out_v, in_v = nums[l - 1], nums[r]
        if win[out_v] == 1:
            del win[out_v]
        else:
            win[out_v] -= 1
        win[in_v] = win.get(in_v, 0) + 1
        cur = cur - out_v + in_v
        dup = sum(c - 1 for c in win.values() if c > 1)
        if dup == 0:
            best = cur if best is None else max(best, cur)
        steps.append(_step(f"window [{l}..{r}]", [("nums", nums, r)],
                           [("l", l, "indigo"), ("r", r, "violet"), ("sum", cur, "amber"),
                            ("dups", dup, "red"), ("best", best if best is not None else "-", "green")],
                           eq=f"drop {out_v}, add {in_v}: sum {cur}, {dup} duplicate(s)"))
    assert best == 15
    return _pack("Distinct fixed window - only an all-unique window is a candidate",
                 f"nums = {nums}, k = {k}", steps, best)


def _distinct_recheck():
    nums, k = DISTINCT_NUMS, DISTINCT_K
    steps, best = [], None
    for l in range(len(nums) - k + 1):
        w = nums[l:l + k]
        dup = k - len(set(w))
        s = sum(w)
        if dup == 0:
            best = s if best is None else max(best, s)
        steps.append(_step(f"window [{l}..{l + k - 1}]", [("nums", nums, l + k - 1)],
                           [("window", w, "violet"), ("dups", dup, "red"),
                            ("sum", s, "amber"), ("best", best if best is not None else "-", "green")],
                           eq=f"re-read {w}: {dup} duplicate(s), sum {s}" + (" -> candidate" if dup == 0 else " -> reject")))
    assert best == 15
    return _pack("Distinct windows, rechecked from scratch - the set is rebuilt every step",
                 f"nums = {nums}, k = {k}", steps, best)


def _cards_mid():
    card, k = [1, 2, 3, 4, 5, 6, 1], 3
    n, w, total = len(card), len(card) - k, sum(card)
    steps, worst = [], None
    for l in range(k + 1):
        win = card[l:l + w]
        mid = sum(win)
        worst = mid if worst is None else min(worst, mid)
        steps.append(_step(f"re-read block [{l}..{l + w - 1}]", [("window", win, w - 1)],
                           [("kept", mid, "amber"), ("cheapest so far", worst, "violet"),
                            ("score", total - worst, "green")],
                           eq=f"sum {win} = {mid}; best score {total} - {worst} = {total - worst}"))
    best = total - worst
    assert best == 12
    return _pack("Enumerate the complement: k + 1 middles, summed fresh",
                 f"cards = {card}, k = {k}, total = {total}", steps, best)


def _cards_slide():
    card, k = [1, 2, 3, 4, 5, 6, 1], 3
    n, w = len(card), len(card) - k
    total = sum(card)
    steps = []
    for l in range(k + 1):
        r = l + w - 1
        mid = sum(card[l:r + 1])
        earned = total - mid
        if l == 0:
            eq = f"keep [{l}..{r}] = {mid} (left pick 0); earned {total} - {mid} = {earned}"
        else:
            eq = (f"drop {card[l - 1]}, add {card[r]}: keep = {mid}; "
                  f"earned {total} - {mid} = {earned}")
        steps.append(_step(f"window [{l}..{r}]", [("cards", card, r)],
                           [("l", l, "indigo"), ("r", r, "violet"), ("kept", mid, "amber"),
                            ("earned", earned, "green")],
                           eq=eq))
    best = max(total - sum(card[l:l + w]) for l in range(k + 1))
    assert best == 12
    return _pack("Maximum Points - slide the kept middle window, earn the complement",
                 "cardPoints = [1, 2, 3, 4, 5, 6, 1], k = 3", steps, best)


def _reduce_middle():
    nums, x = [1, 1, 4, 2, 3], 5
    need = sum(nums) - x                    # the kept middle must sum to exactly this
    l = s = 0
    best, best_l = -math.inf, 0
    steps = []
    for r, v in enumerate(nums):
        s += v
        tag = ""
        while s > need:                     # positives: only dropping the front can help
            s -= nums[l]
            l += 1
            tag = " (shrink)"
        hit = s == need
        if hit and r - l + 1 > best:
            best, best_l = r - l + 1, l
        head = f"overshot {need}, dropped the front; " if tag else ""
        steps.append(_step(f"r = {r}{tag}", [("nums", nums, r)],
                           [("l", l, "indigo"), ("r", r, "violet"), ("sum", s, "amber"),
                            ("longest", best if best > -math.inf else "-", "green")],
                           eq=head + f"middle [{l}..{r}] sums to {s}"
                           + (f" = {need}: candidate, length {r - l + 1}" if hit else f" < {need}: grow")))
    assert best == 3 and len(nums) - best == 2
    kept = nums[best_l:best_l + best]
    assert kept == [1, 1, 4] and sum(kept) == need
    steps.append(_step("remove the ends", [("nums", nums, None)],
                       [("kept", kept, "violet"), ("ops", len(nums) - int(best), "green")],
                       eq=f"keep {kept} (sum {need}); the other {len(nums) - int(best)} elements are the operations"))
    return _pack("Reduce X to Zero - longest middle summing to total - x",
                 "nums = [1, 1, 4, 2, 3], x = 5; keep a middle summing to 11 - 5 = 6", steps, len(nums) - int(best))


def _reduce_pmap():
    nums, x = [1, 1, 4, 2, 3], 5
    need = sum(nums) - x
    first = {0: -1}
    P = 0
    best = -1
    steps = []
    for r, v in enumerate(nums):
        P += v
        if P - need in first:
            cand = r - first[P - need]
            best = max(best, cand)
            eq = f"prefix {P} minus prefix {P - need} at index {first[P - need]}: middle length {cand}"
        else:
            eq = f"prefix {P}: no stored prefix equals {P} - {need} = {P - need}"
        first.setdefault(P, r)
        steps.append(_step(f"r = {r}", [("stored prefixes", list(first), None)],
                           [("P", P, "indigo"), ("P - need", P - need, "violet"),
                            ("best", best if best >= 0 else "-", "green")],
                           eq=eq))
    assert best == 3
    return _pack("Reduce X with a prefix map - works even without positivity, at O(n) space",
                 "nums = [1, 1, 4, 2, 3], x = 5; keep a middle summing to 6", steps, len(nums) - best)


def _k_atmost():
    nums, k = [1, 2, 1, 2, 3], 2
    cnt = Counter()
    l = n_at = 0
    steps = []
    for r, x in enumerate(nums):
        cnt[x] += 1
        while len(cnt) > k:
            cnt[nums[l]] -= 1
            if not cnt[nums[l]]:
                del cnt[nums[l]]
            l += 1
        n_at += r - l + 1
        steps.append(_step(f"r = {r}", [("nums", nums, r)],
                           [("l", l, "indigo"), ("r", r, "violet"),
                            ("distinct", len(cnt), "amber"), ("at_most", n_at, "green")],
                           eq=f"window [{l}..{r}] has {len(cnt)} distinct <= {k}: "
                              f"{r - l + 1} new subarrays ending at {r}"))
    cnt1, l1, n_less = Counter(), 0, 0          # the second pass: at most k - 1 distinct
    for r, x in enumerate(nums):
        cnt1[x] += 1
        while len(cnt1) > k - 1:
            cnt1[nums[l1]] -= 1
            if not cnt1[nums[l1]]:
                del cnt1[nums[l1]]
            l1 += 1
        n_less += r - l1 + 1
    assert (n_at, n_less) == (12, 5)
    steps.append(_step("subtract", [("nums", nums, None)],
                       [(f"at_most({k})", n_at, "amber"), (f"at_most({k - 1})", n_less, "red")],
                       eq=f"exactly {k} distinct = at_most({k}) - at_most({k - 1}) = {n_at} - {n_less}"))
    return _pack("Subarrays with K Different Integers - count 'at most', then subtract",
                 "nums = [1, 2, 1, 2, 3], k = 2", steps, n_at - n_less)


def _kdist_perleft():
    nums, k = [1, 2, 1, 2, 3], 2
    n = len(nums)
    steps, total = [], 0
    for l in range(n):
        seen = set()
        first_at_k = None
        r = l - 1
        while r + 1 < n:
            nxt = set(seen) | {nums[r + 1]}
            if len(nxt) > k:
                break
            seen, r = nxt, r + 1
            if len(seen) == k and first_at_k is None:
                first_at_k = r
        c = (r - first_at_k + 1) if first_at_k is not None else 0
        total += c
        steps.append(_step(f"left l = {l}", [("nums", nums, r)],
                           [("l", l, "indigo"), ("first exact", first_at_k if first_at_k is not None else "-", "amber"),
                            ("furthest", r, "violet"), ("count", total, "green")],
                           eq=f"from {l}, k distinct first holds at r = {first_at_k}, "
                              f"still holds through r = {r}: +{c}"))
    assert total == 7
    return _pack("Exactly K distinct, per left end - no reuse between starts",
                 "nums = [1, 2, 1, 2, 3], k = 2", steps, total)


def _maxfreq_bs():
    nums, k = [1, 3, 2, 3, 3], 2
    mx = max(nums)
    pos = [i for i, x in enumerate(nums) if x == mx]
    n = len(nums)
    steps, total = [], 0
    for l in range(n):
        t = bisect.bisect_left(pos, l)      # first max at or after l
        if t + k - 1 < len(pos):
            j = pos[t + k - 1]              # the k-th max counted from l
            total += n - j
            eq = (f"{k}-th max from {l} sits at pos[{t + k - 1}] = {j}: every right end "
                  f">= {j} works -> +{n - j}")
        else:
            eq = f"fewer than {k} maxes remain from {l}: contributes nothing"
        steps.append(_step(f"left l = {l}", [("positions of max", pos, t + k - 1 if t + k - 1 < len(pos) else None)],
                           [("l", l, "indigo"), ("k-th max at", pos[t + k - 1] if t + k - 1 < len(pos) else "-", "violet"),
                            ("count", total, "green")],
                           eq=eq))
    assert total == 6
    return _pack("Max at least K times - per left end, bisect the k-th max's position",
                 "nums = [1, 3, 2, 3, 3], k = 2; positions of 3: [1, 3, 4]", steps, total)


def _maxfreq_batch():
    nums, k = [1, 3, 2, 3, 3], 2
    mx, n = max(nums), len(nums)
    l = hits = total = 0
    steps = []
    for r, x in enumerate(nums):
        hits += x == mx
        added = []
        while hits >= k:                     # good: every end r..n-1 works for start l
            total += n - r
            added.append(l)
            hits -= nums[l] == mx
            l += 1
        if added:
            note = (f"start(s) {', '.join(map(str, added))} first turn good at r = {r}: "
                    f"+{n - r} each; l moves to {l}")
        else:
            note = f"{hits} copy(ies) of {mx} in [{l}..{r}], fewer than {k}: grow"
        steps.append(_step(f"r = {r}", [("nums", nums, r)],
                           [("l", l, "indigo"), ("r", r, "violet"), ("hits", hits, "amber"),
                            ("count", total, "green")],
                           eq=note))
    assert total == 6
    return _pack("Max appears at least K times - batch-add n - r",
                 "nums = [1, 3, 2, 3, 3], k = 2", steps, total)


_ALL = None


def get(problem, form):
    """Return ``(title, input line, steps, final)`` for one solution form."""
    global _ALL
    if _ALL is None:
        _ALL = {
            ("avg", "fixed"): _avg_fixed(),
            ("avg", "prefix"): _avg_prefix(),
            ("ones", "grow"): _ones_grow(),
            ("ones", "bs"): _ones_bs(),
            ("ones", "deque"): _ones_deque(),
            ("books", "grow"): _books_grow(),
            ("books", "bs"): _books_bs(),
            ("mindist", "shrink"): _mindist_shrink(),
            ("mindist", "bs"): _mindist_bs(),
            ("product", "count"): _product_count(),
            ("product", "perleft"): _product_perleft(),
            ("distinct", "fixed"): _distinct_fixed(),
            ("distinct", "recheck"): _distinct_recheck(),
            ("cards", "mid"): _cards_mid(),
            ("cards", "slide"): _cards_slide(),
            ("reduce", "middle"): _reduce_middle(),
            ("reduce", "pmap"): _reduce_pmap(),
            ("kdist", "atmost"): _k_atmost(),
            ("kdist", "perleft"): _kdist_perleft(),
            ("maxfreq", "bs"): _maxfreq_bs(),
            ("maxfreq", "batch"): _maxfreq_batch(),
        }
    return _ALL[(problem, form)]
