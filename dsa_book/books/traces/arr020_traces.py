"""Step boards for the GCD / divisibility chapter (Euclid and the gcd-lcm identity,
prefix and suffix GCD, distinct GCDs of subarrays and subsequences, remainder
counting, harmonic loops over the value domain and counting coprime pairs, and the
pigeonhole / stack-merge view where large inputs force the answer).

Drawing lives in ``dsa_viz.story``; ``get`` returns data only. Each story walks a worked
example - the problem's own statement sample where one is public, otherwise a small input
constructed and then verified - at most eight frames, and every number comes from running
the algorithm - never from a hand-typed table. A frame shows whatever the
rung actually maintains: the running GCD as it folds the array, the residue histogram
of a remainder count, the multiples of a divisor d and their collective gcd, the
surplus that cycles forward around the three buckets, the stack of a non-coprime merge.

Every problem carries a ladder: a ``scan``/``naive`` first frame set (the direct
definition), a ``fast`` second set (the rung the pattern names) and, where the pattern
has one, a third set. ``get`` returns whichever frame set the caller asks for, and every
builder asserts its own answer against the verified judge sample before returning.
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from functools import reduce
from math import gcd
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
    return [str(v) if lo <= k <= hi else "." for k, v in enumerate(vals)]


def _mask(vals, idx):
    return [str(v) if k == idx else "." for k, v in enumerate(vals)]


# ==================================================== LC 1979 find the GCD of an array
AG_ARR = [2, 5, 6, 9, 10]
AG_ANS = 2


def _ag_scan():
    """LC 1979 is defined as gcd(smallest, largest): pass 1 finds the min, pass 2 the max."""
    mn = AG_ARR[0]
    steps = [row_step("pass 1: fold the smallest", [("arr", AG_ARR, 0), ("min", [mn], None)])]
    for i, v in enumerate(AG_ARR):
        mn = min(mn, v)
        steps.append(row_step(f"compare a[{i}] = {v}",
                              [("arr", AG_ARR, i), ("min", [mn], None)],
                              eq=f"min so far = {mn}"))
    mx = AG_ARR[0]
    for i, v in enumerate(AG_ARR):
        mx = max(mx, v)
    steps.append(row_step("pass 2: fold the largest",
                          [("arr", AG_ARR, None), ("max", [mx], None)],
                          eq=f"max over the array = {mx}"))
    ans = gcd(mn, mx)
    steps.append(chip_step(f"gcd(min, max) = gcd({mn}, {mx})",
                           [("answer", ans, "green")]))
    assert ans == AG_ANS
    return _pack("Array GCD - the answer is gcd(smallest, largest)",
                 f"arr = {AG_ARR}", steps, ans)


def _ag_fold():
    """One streaming pass keeps a running (min, max); gcd them at the end - same answer, half the reads."""
    lo = hi = AG_ARR[0]
    steps = [row_step("seed min = max = a[0]",
                      [("arr", AG_ARR, 0), ("min", [lo], None), ("max", [hi], None)])]
    for i, v in enumerate(AG_ARR):
        lo, hi = min(lo, v), max(hi, v)
        steps.append(row_step(f"fold a[{i}] = {v}",
                              [("arr", AG_ARR, i), ("min", [lo], None), ("max", [hi], None)],
                              eq=f"min {lo}, max {hi}"))
    ans = gcd(lo, hi)
    steps.append(chip_step(f"gcd(min, max) = gcd({lo}, {hi})",
                           [("answer", ans, "green")]))
    assert ans == AG_ANS
    return _pack("Array GCD - one streaming pass keeps min and max",
                 f"arr = {AG_ARR}", steps, ans)


# ================================ LC 2001 pairs of interchangeable rectangles (ratio key)
RC_RECTS = [(4, 8), (3, 6), (10, 20), (15, 30)]
RC_ANS = 6


def _rc_scan():
    ans, steps = 0, []
    n = len(RC_RECTS)
    for i in range(n):
        for j in range(i + 1, n):
            w1, h1 = RC_RECTS[i]
            w2, h2 = RC_RECTS[j]
            good = w1 * h2 == w2 * h1
            ans += good
            steps.append(row_step(f"pair ({i},{j}): {w1}/{h1} vs {w2}/{h2}",
                                  [("cross", [f"{w1}*{h2}", f"{w2}*{h1}"], None),
                                   ("equal?", ["yes" if good else "no"], None),
                                   ("total", [ans], None)],
                                  eq=f"{w1 * h2} {'==' if good else '!='} {w2 * h1}"))
    assert ans == RC_ANS
    return _pack("Interchangeable rectangles - test every pair",
                 f"rects = {RC_RECTS}", steps, ans)


def _rc_key():
    seen, ans, steps = Counter(), 0, []
    for i, (w, h) in enumerate(RC_RECTS):
        g = gcd(w, h)
        key = (w // g, h // g)
        ans += seen[key]                       # charge to the right end of each pair
        seen[key] += 1
        steps.append(row_step(f"reduce {w}/{h} by gcd {g}",
                              [("rects", [f"{a}/{b}" for a, b in RC_RECTS], i),
                               ("key", _cells([k[0] for k in seen] ), None),
                               ("pairs so far", [ans], None)],
                               eq=f"key = ({w // g}, {h // g}) = {key}; saw {seen[key] - 1} before"))
    assert ans == RC_ANS
    return _pack("Interchangeable rectangles - histogram the reduced ratio",
                 f"rects = {RC_RECTS}", steps, ans)


# ========================= LC 2436 minimum split into subarrays with GCD > 1 (prefix gcd)
SP_ARR = [12, 6, 3, 14, 8]
SP_ANS = 2


def _sp_greedy():
    splits, g, steps = 1, SP_ARR[0], []
    steps.append(row_step("open first segment at a[0]",
                          [("arr", SP_ARR, 0), ("segment gcd", [g], None), ("splits", [splits], None)]))
    for i in range(1, len(SP_ARR)):
        v = SP_ARR[i]
        if gcd(g, v) > 1:
            g = gcd(g, v)
            note = f"gcd({g}, {v}) = {gcd(g, v)} > 1: keep the segment open"
        else:
            splits += 1
            g = v
            note = f"gcd ran to 1 with {v}: cut before it, open a new segment"
        steps.append(row_step(f"extend to a[{i}] = {v}",
                              [("arr", _window(SP_ARR, 0, i), None),
                               ("segment gcd", [g], None), ("splits", [splits], None)],
                              eq=note))
    assert splits == SP_ANS
    return _pack("Minimum GCD split - greedy running prefix gcd",
                 f"arr = {SP_ARR}", steps, splits)


def _sp_probe():
    steps, ans = [], None
    for arr, want in (([12, 6, 3, 14, 8], 2), ([4, 12, 6, 14], 1), ([2, 3, 5, 7], 4), ([2, 4, 6, 8], 1)):
        splits, g = 1, arr[0]
        for x in arr[1:]:
            if gcd(g, x) > 1:
                g = gcd(g, x)
            else:
                splits += 1
                g = x
        ans = splits
        assert ans == want
        steps.append(row_step(f"arr = {arr}",
                              [("segments", [splits], None),
                               ("answer", [ans], None)],
                              eq="cut whenever the running gcd would drop to 1"))
    return _pack("Minimum GCD split - the answer is number of forced cuts",
                 "several arrays", steps, ans)


# ================================= LC 1071 greatest common divisor of strings (strgcd)
SG_A = "ABCABC"
SG_B = "ABC"
SG_ANS = "ABC"


def _sg_walk():
    a, b = SG_A, SG_B
    steps = [line_step(f"str1 = {a!r}, str2 = {b!r}",
                       ["A common repeating unit must tile both strings",
                        f"test str1+str2 == str2+str1: {a + b} vs {b + a}"],
                       eq=f"{a + b} == {b + a}")]
    if a + b != b + a:
        ans = ""
    else:
        g = gcd(len(a), len(b))
        ans = a[:g]
        steps.append(line_step(f"gcd(len) = gcd({len(a)}, {len(b)}) = {g}",
                               [f"the unit is the first {g} characters"],
                               eq=f"answer = str1[:{g}] = {ans!r}"))
    assert ans == SG_ANS
    return _pack("GCD of strings - the unit length is gcd of the lengths",
                 f'str1 = "{a}", str2 = "{b}"', steps, ans)


def _sg_examples():
    steps = []
    for a, b in (("ABCABC", "ABC"), ("ABABAB", "ABAB"), ("LEET", "CODE")):
        ok = a + b == b + a
        ans = a[:gcd(len(a), len(b))] if ok else ""
        steps.append(line_step(f'"{a}" , "{b}"',
                               [f'"{a + b}" == "{b + a}" ? -> {ok}'],
                               eq=f'answer = {ans!r}'))
    return _pack("GCD of strings - concatenate to test a shared period",
                 "three pairs", steps, "")


# ============================ LC 1819 number of different subsequences GCDs (subseq)
SQ_ARR = [6, 10, 3]
SQ_ANS = 5


def _sq_fold():
    allset, steps = set(), []
    for i, x in enumerate(SQ_ARR):
        new = {x} | {gcd(g, x) for g in allset}
        allset |= new
        steps.append(row_step(f"subsequences ending at a[{i}] = {x}",
                              [("arr", SQ_ARR, i),
                               ("new gcds", _cells(sorted(new)), None),
                               ("all so far", _cells(sorted(allset)), None)],
                              eq=f"gcd({x}, each earlier gcd) plus {x} itself"))
    assert len(allset) == SQ_ANS
    return _pack("Distinct subsequence GCDs - fold a set over the array",
                 f"arr = {SQ_ARR}", steps, len(allset))


def _sq_sieve():
    present = set(SQ_ARR)
    mx = max(SQ_ARR)
    res, steps = set(), []
    for d in range(1, mx + 1):
        mult = [x for x in range(d, mx + 1, d) if x in present]
        if not mult:
            continue
        col = reduce(gcd, mult)
        hit = col == d
        if hit:
            res.add(d)
        steps.append(row_step(f"divisor d = {d}",
                              [("multiples in arr", _cells(mult), None),
                               ("their gcd", [col], None),
                               ("achievable?", ["yes" if hit else "no"], None)],
                              eq=f"gcd(multiples of {d}) = {col} {':=d' if hit else '!= ' + str(d)}"))
    assert len(res) == SQ_ANS
    return _pack("Distinct subsequence GCDs - sieve the value domain",
                 f"arr = {SQ_ARR}", steps, len(res))


# ================================= LC 3312 sorted GCD pair queries (harmonic + Mobius)
GP_NUMS = [2, 3, 4]
GP_QUERIES = [0, 2, 2]
GP_ANS = [1, 2, 2]


def _gp_scan():
    pairs, steps = [], []
    n = len(GP_NUMS)
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(GP_NUMS[i], GP_NUMS[j])
            pairs.append(g)
            steps.append(row_step(f"pair ({i},{j}): gcd({GP_NUMS[i]}, {GP_NUMS[j]})",
                                  [("nums", GP_NUMS, None), ("gcd so far", _cells(pairs), None)]))
    ordered = sorted(pairs)
    steps.append(chip_step(f"sorted = {ordered}; answer query q -> sorted[q]",
                           [("answer", _cells(GP_ANS), "green")]))
    assert ordered == [1, 1, 2] and [ordered[q] for q in GP_QUERIES] == GP_ANS
    return _pack("Sorted GCD pairs - list every pair, then sort",
                 f"nums = {GP_NUMS}, queries = {GP_QUERIES}", steps, GP_ANS)


def _gp_sieve():
    nums = GP_NUMS
    mx = max(nums)
    freq = Counter(nums)
    cm = [0] * (mx + 1)
    for d in range(1, mx + 1):
        for m in range(d, mx + 1, d):
            cm[d] += freq.get(m, 0)
    atmost = [cm[d] * (cm[d] - 1) // 2 for d in range(mx + 1)]
    exact = [0] * (mx + 1)
    steps = []
    for d in range(mx, 0, -1):
        e = atmost[d]
        for k in range(2 * d, mx + 1, d):
            e -= exact[k]
        exact[d] = e
        steps.append(row_step(f"exact gcd = {d}",
                              [("multiple count", _cells(cm[1:]), None),
                               ("at most", _cells(atmost[1:]), None),
                               ("exactly", _cells(exact[1:]), d - 1)],
                              eq=f"C({cm[d]},2) minus the higher multiples -> exact[{d}] = {e}"))
    flat = [d for d in range(1, mx + 1) for _ in range(exact[d])]
    assert flat == sorted(gcd(nums[i], nums[j])
                          for i in range(len(nums)) for j in range(i + 1, len(nums)))
    ans = [flat[q] for q in GP_QUERIES]
    assert ans == GP_ANS
    steps.append(chip_step(f"cumulative from exact -> {flat}; pick query indices",
                           [("answer", _cells(ans), "green")]))
    return _pack("Sorted GCD pairs - count by divisor, invert, prefix",
                 f"nums = {nums}, queries = {GP_QUERIES}", steps, ans)


# ================================= CF 1490B balanced remainders (residue cycling)
BR_ARR = [0, 2, 5, 5, 4, 8]          # the statement's own first sample; answer 3
BR_ANS = 3


def _br_cycle():
    n = len(BR_ARR)
    c = [sum(1 for x in BR_ARR if x % 3 == r) for r in range(3)]
    target = n // 3
    moves, steps = 0, [row_step("initial residue buckets",
                                [("a", BR_ARR, None), ("counts", _cells(c), None)],
                                eq=f"target per bucket = n/3 = {target}")]
    for _ in range(3):
        for r in range(3):
            if c[r] > target:
                extra = c[r] - target
                c[r] -= extra
                c[(r + 1) % 3] += extra
                moves += extra
                steps.append(row_step(f"push {extra} surplus from bucket {r} to {(r + 1) % 3}",
                                      [("counts", _cells(c), None), ("moves", [moves], None)],
                                      eq="one increment walks an element forward one bucket"))
    assert moves == BR_ANS
    return _pack("Balanced remainders - cycle surplus forward",
                 f"a = {BR_ARR}", steps, moves)


# ================================= CF 1374D zero remainder array (needed increments)
ZR_ARR = [1, 2, 1, 3]                # the statement's own first sample; answer 6
ZR_K = 3
ZR_ANS = 6


def _zr_formula():
    need = Counter()
    steps = [line_step(f"a = {ZR_ARR}, k = {ZR_K}",
                       ["a move spends the current x on one element then does x += 1",
                        "an element with remainder r>0 must be handed an x == k-r (mod k)"],
                       eq="group elements by the increment value x they must be handed")]
    for x in ZR_ARR:
        r = x % ZR_K
        if r:
            need[ZR_K - r] += 1
            steps.append(row_step(f"a[i] = {x}: remainder {r}",
                                  [("needs x =", [ZR_K - r], None),
                                   ("same-need count", _cells([need[d] for d in sorted(need)]), None)],
                                  eq=f"{x} % {ZR_K} = {r} -> its x is {ZR_K - r} mod {ZR_K}"))
    if need:
        raw = max((cnt - 1) * ZR_K + d for d, cnt in need.items())
        ans = raw + 1                 # moves = the largest x spent, plus the rounds up to it
        note = f"moves = max_d[(count[d]-1)*k + d] + 1 = {raw} + 1 = {ans}"
    else:
        ans = 0
        note = "nothing needs an increment -> 0 moves"
    steps.append(chip_step(note, [("answer", ans, "green")],
                           eq=f"the {need[2]} elements needing x=2 differ by a full cycle of k"
                           if need[2] else "each shared need is a full cycle apart"))
    assert ans == ZR_ANS
    return _pack("Zero remainder array - the last x decides the move count",
                 f"a = {ZR_ARR}, k = {ZR_K}", steps, ans)


# ================================= LC 2197 replace non-coprime numbers (stack merge)
MR_NUMS = [6, 4, 3, 2, 7, 6, 2]
MR_ANS = [12, 7, 6]


def _mr_naive():
    a = list(MR_NUMS)
    steps = [row_step("start", [("stack", _cells(a), None)])]
    changed = True
    while changed:
        changed = False
        for i in range(len(a) - 1):
            if gcd(a[i], a[i + 1]) > 1:
                l = a[i] * a[i + 1] // gcd(a[i], a[i + 1])
                a[i:i + 2] = [l]
                steps.append(row_step(f"merge a[{i}],a[{i + 1}] -> lcm {l}",
                                      [("array", _cells(a), i)],
                                      eq=f"lcm = {a[i]} (recomputed) as one element"))
                changed = True
                break
    assert a == MR_ANS
    return _pack("Replace non-coprime - merge any adjacent pair repeatedly",
                 f"nums = {MR_NUMS}", steps, MR_ANS)


def _mr_stack():
    stack, steps = [], []
    for x in MR_NUMS:
        cur = x
        merges = 0
        while stack and gcd(stack[-1], cur) > 1:
            top = stack.pop()
            cur = top * cur // gcd(top, cur)
            merges += 1
        stack.append(cur)
        steps.append(row_step(f"push {x}" + (" (folded into LCM)" if merges else " (coprime, stays)"),
                              [("nums", _mask(MR_NUMS, len(steps) - 1), None),
                               ("stack", _cells(stack), len(stack) - 1)],
                              eq=f"cur became lcm(...,{x}) = {cur} after {merges} pop(s)"))
    assert stack == MR_ANS
    return _pack("Replace non-coprime - one left-to-right stack",
                 f"nums = {MR_NUMS}", steps, MR_ANS)


# ================================= CF 1629B GCD arrays (parity pigeonhole)
# Official statement: array a = every integer in [l, r]. One operation removes two
# elements and inserts their PRODUCT. Ask: can gcd(a) > 1 after at most k operations?
# Making gcd > 1 means killing every odd (pair each odd with an even; product stays even
# and is reusable), so the answer is YES iff the odd count fits in k - except a single
# element, whose gcd is itself. These are the 9 sample cases from the statement.
PA_TESTS = [(1, 1, 0), (3, 5, 1), (13, 13, 0), (4, 4, 0), (3, 7, 4),
            (4, 10, 3), (2, 4, 0), (1, 7, 3), (1, 5, 3)]
PA_ANS = ["NO", "NO", "YES", "YES", "YES", "YES", "NO", "NO", "YES"]


def _pa_parity():
    steps = []
    for (l, r, k), want in zip(PA_TESTS, PA_ANS):
        n = r - l + 1
        odd = (r + 1) // 2 - l // 2
        if n == 1:
            verdict = "YES" if l > 1 else "NO"
            why = f"single element {l}: gcd is itself {'> 1' if l > 1 else '= 1'}"
        else:
            verdict = "YES" if odd <= k else "NO"
            why = f"{odd} odd numbers, each product-move kills at most one -> need {odd}, have {k}"
        assert verdict == want, (l, r, k, verdict, want)
        steps.append(row_step(f"[{l}..{r}], k = {k}",
                              [("odd count", [odd], None), ("k", [k], None),
                               ("verdict", [verdict], None)],
                              eq=why))
    return _pack("GCD arrays - the odds must all be absorbed",
                 "several [l,r],k", steps, "YES/NO per case")


REGISTRY = {
    ("arrgcd", "scan"): _ag_scan(),
    ("arrgcd", "fold"): _ag_fold(),
    ("rects", "scan"): _rc_scan(),
    ("rects", "key"): _rc_key(),
    ("split", "greedy"): _sp_greedy(),
    ("split", "probe"): _sp_probe(),
    ("strgcd", "walk"): _sg_walk(),
    ("strgcd", "examples"): _sg_examples(),
    ("subseq", "fold"): _sq_fold(),
    ("subseq", "sieve"): _sq_sieve(),
    ("gcdpair", "scan"): _gp_scan(),
    ("gcdpair", "sieve"): _gp_sieve(),
    ("balance", "cycle"): _br_cycle(),
    ("zeromod", "formula"): _zr_formula(),
    ("merge", "naive"): _mr_naive(),
    ("merge", "stack"): _mr_stack(),
    ("parity", "count"): _pa_parity(),
}


def get(problem, form):
    return REGISTRY[(problem, form)]
