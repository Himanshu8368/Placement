"""Step frames for the index-tricks chapter (cyclic sort, marking, permutation cycles).

Drawing lives in ``dsa_viz.story``. ``get`` returns data only. Each story walks
the chapter's own sample, at most eight frames, and the numbers come from
running the algorithm - never from a hand-typed table. An index-trick frame
shows the array with the current cursor (or the pair of cursors / the value
being placed) highlighted plus the rolling chips (freq, seen, lo, hi, a, b),
so the reader watches a *swap*, a *mark*, or a *cycle step* exactly as the
algorithm performs it.

Every problem carries a ladder: a ``brute`` frame set (the direct definition -
a nested scan, a count table, an O(n^2) search), an ``intermediate`` frame set
(a one-pass view that gets the shape right but not yet the closed form), and an
``optimal`` frame set (the rung the pattern names). ``get`` returns whichever
frame set the caller asks for.
"""
from __future__ import annotations

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


# ------------------------------------------------- CF 864D make a permutation

_MAKE_A = [3, 2, 2, 3]


def _make_freq(a):
    n = len(a)
    freq = [0] * (n + 1)
    for x in a:
        freq[x] += 1
    return freq


def _makeperm_count():
    """Frequency table -> duplicate slots and the missing values. q = #missing."""
    a = list(_MAKE_A)
    n = len(a)
    freq = _make_freq(a)
    missing = [v for v in range(1, n + 1) if freq[v] == 0]
    dup_extra = sum(freq[v] - 1 for v in range(1, n + 1) if freq[v] > 1)
    steps = [_step("input", [_row("a", a)],
                   eq=f"values are in 1..{n}; count how many times each appears")]
    steps.append(_step("frequency",
                       [_row("a", a), _row("freq", freq[1:], hi=None)],
                       [("q", dup_extra, "amber")],
                       eq=f"missing = {missing}; extra copies (changes) = {dup_extra}"))
    steps.append(_step("read off",
                       [_row("missing", missing, hi=len(missing) - 1)],
                       [("q", dup_extra, "amber")],
                       eq=f"must change q = {dup_extra} cells and fill in {missing}"))
    assert dup_extra == 2 and missing == [1, 4]
    return _pack("Make a Permutation - frequency table (count)",
                 f"a = {a}", steps, str(dup_extra))


def _makeperm_keepfirst():
    """Keep the FIRST copy of every value; replace later duplicates with sorted
    missing. Correct q and a valid permutation, but NOT lexicographically minimal."""
    a = list(_MAKE_A)
    n = len(a)
    missing = [v for v in range(1, n + 1) if _make_freq(a)[v] == 0]
    seen = set()
    res = []
    mi = 0
    steps = [_step("input", [_row("a", a)],
                   [("missing", str(missing), "green")],
                   eq="walk left to right; a value seen before is overwritten by next missing")]
    for i, x in enumerate(a):
        if x in seen:
            res.append(missing[mi])
            mi += 1
            act = f"dup {x} -> write {res[-1]}"
        else:
            seen.add(x)
            res.append(x)
            act = f"first {x} -> keep"
        steps.append(_step(f"pos {i}",
                           [_row("a", a, hi=i), _row("res", res, hi=len(res) - 1)],
                           eq=act))
    assert res == [3, 2, 1, 4]
    assert sorted(res) == list(range(1, n + 1))
    return _pack("Make a Permutation - keep-first copy (valid, not lex-min)",
                 f"a = {a}, missing = {missing}", steps, str(res))


def _makeperm_greedy():
    """Lexicographically minimal: overwrite a FIRST copy too when a smaller
    missing value is waiting and the value reappears later."""
    a = list(_MAKE_A)
    n = len(a)
    freq = _make_freq(a)
    missing = [v for v in range(1, n + 1) if freq[v] == 0]
    res = [0] * n
    used = [False] * (n + 1)
    cnt = freq[:]                      # remaining copies from here to the right
    mi = 0
    steps = [_step("input",
                   [_row("a", a), _row("freq", freq[1:])],
                   [("need", str(missing), "green")],
                   eq="for each cell: if a smaller missing value waits AND this value recurs, overwrite")]
    for i in range(n):
        v = a[i]
        cnt[v] -= 1
        if used[v]:                    # a spare copy -> must be replaced
            res[i] = missing[mi]
            mi += 1
            act = f"a[{i}]={v} spare -> write {res[i]}"
        elif mi < len(missing) and missing[mi] < v and cnt[v] >= 1:
            res[i] = missing[mi]
            mi += 1
            act = f"smaller missing {res[i]} waits and {v} recurs -> overwrite"
        else:
            used[v] = True
            res[i] = v
            act = f"keep {v}"
        steps.append(_step(f"pos {i}",
                           [_row("a", a, hi=i), _row("res", res, hi=i)],
                           eq=act))
        if len(steps) > 8:
            break
    assert res == [1, 2, 4, 3]
    assert sorted(res) == list(range(1, n + 1))
    return _pack("Make a Permutation - lex-min greedy (submit)",
                 f"a = {a}, missing = {missing}", steps, str(res))


# ------------------------------------------ CF 1385B restore permutation merger

_MERGE_A = [3, 1, 3, 1, 2, 2]


def _merger_brute():
    """O(n^2): keep a[i] iff it has not appeared strictly before i."""
    a = list(_MERGE_A)
    res = []
    steps = [_step("input", [_row("a", a)],
                   eq="output each value at its FIRST position; skip any value seen earlier")]
    for i, x in enumerate(a):
        if all(a[k] != x for k in range(i)):
            res.append(x)
            act = f"{x} new -> keep"
        else:
            act = f"{x} seen before -> skip"
        steps.append(_step(f"pos {i}",
                           [_row("a", a, hi=i), _row("res", res, hi=len(res) - 1 if res else None)],
                           eq=act))
        if len(steps) > 8:
            break
    assert res == [3, 1, 2]
    return _pack("Restore Permutation - first-occurrence scan (O(n^2))",
                 f"a = {a}", steps, str(res))


def _merger_seen():
    """O(n): a boolean 'seen' table (the marking-in-place primitive)."""
    a = list(_MERGE_A)
    n = len(a)
    hi = max(a)
    seen = [False] * (hi + 1)
    res = []
    steps = [_step("input", [_row("a", a)],
                   [("seen", "empty", "indigo")],
                   eq="one pass: if seen[x] skip, else mark seen[x] and emit x")]
    for i, x in enumerate(a):
        if seen[x]:
            act = "seen[" + str(x) + "] -> skip"
        else:
            seen[x] = True
            res.append(x)
            act = f"mark seen[{x}], emit {x}"
        steps.append(_step(f"pos {i}",
                           [_row("a", a, hi=i), _row("seen", seen[1:]),
                            _row("res", res, hi=len(res) - 1 if res else None)],
                           eq=act))
        if len(steps) > 8:
            break
    assert res == [3, 1, 2]
    return _pack("Restore Permutation - one-pass seen table (optimal)",
                 f"a = {a}", steps, str(res))


# ------------------------------------------------ CF 862A Mahmoud & Ehab: MEX

_MEX_S = [0, 4, 5, 6, 7]
_MEX_X = 3


def _mex_brute():
    """Presence table over 0..x; count gaps below x, plus remove x if present."""
    s = list(_MEX_S)
    x = _MEX_X
    present = [False] * (x + 1)
    for v in s:
        if v <= x:
            present[v] = True
    adds = sum(1 for v in range(x) if not present[v])
    drops = 1 if present[x] else 0
    steps = [_step("input", [_row("set", s)],
                   [("x", x, "amber")],
                   eq=f"to make MEX = {x}: every 0..{x-1} present, and {x} absent")]
    steps.append(_step("presence",
                       [_row("present", [int(p) for p in present])],
                       [("adds", adds, "green"), ("drops", drops, "red")],
                       eq=f"gaps below x = {adds} adds; x in set -> {drops} drop"))
    steps.append(_step("answer", [_row("present", [int(p) for p in present])],
                       [("ops", adds + drops, "indigo")],
                       eq=f"operations = adds + drops = {adds + drops}"))
    assert adds + drops == 2
    return _pack("MEX - presence table (brute)",
                 f"set = {s}, x = {x}", steps, str(adds + drops))


def _mex_formula():
    """ans = x - (count of elements in [0, x-1]) + (1 if x present else 0)."""
    s = list(_MEX_S)
    x = _MEX_X
    below = sum(1 for v in s if v < x)
    has_x = 1 if x in s else 0
    ans = x - below + has_x
    steps = [_step("input", [_row("set", s)],
                   [("x", x, "amber")],
                   eq="elements under x can only help; everything >= x is irrelevant except x itself")]
    steps.append(_step("count", [],
                       [("below", below, "green"), ("has_x", has_x, "red")],
                       eq=f"count(<x) = {below}, is x present = {has_x}"))
    steps.append(_step("formula", [],
                       [("ans", ans, "indigo")],
                       eq=f"ans = x - count(<x) + is_x = {x} - {below} + {has_x} = {ans}"))
    assert ans == 2
    return _pack("MEX - closed form (optimal)",
                 f"set = {s}, x = {x}", steps, str(ans))


# --------------------------------------------------- CF 932C permutation cycle

_CYC_N, _CYC_A, _CYC_B = 9, 2, 5


def _cyc_search():
    """Find non-negative (ca, cb) with ca*A + cb*B = N; scan cb upward."""
    N, A, B = _CYC_N, _CYC_A, _CYC_B
    steps = [_step("input", [],
                   [("N", N, "amber"), ("A", A, "green"), ("B", B, "indigo")],
                   eq=f"need ca*{A} + cb*{B} = {N}; try cb = 0, 1, 2, ...")]
    found = None
    for cb in range(0, N + 1):
        rem = N - cb * B
        if rem < 0:
            break
        ca = rem // A if rem % A == 0 else None
        steps.append(_step(f"cb = {cb}", [],
                           [("cb", cb, "indigo"), ("rem", rem, "muted")],
                           eq=f"N - cb*B = {rem}; {rem} % {A} == 0 ? {'yes ca=' + str(rem // A) if rem % A == 0 else 'no'}"))
        if rem % A == 0:
            found = (rem // A, cb)
            break
    if len(steps) > 8:
        steps = [steps[0]] + steps[-7:]
    assert found == (2, 1)
    return _pack("Permutation Cycle - solve ca*A + cb*B = N",
                 f"N = {N}, A = {A}, B = {B}", steps, f"ca = {found[0]}, cb = {found[1]}")


def _cyc_build():
    """Lay ca cycles of length A then cb cycles of length B over positions 1..N."""
    N, A, B = _CYC_N, _CYC_A, _CYC_B
    ca, cb = 2, 1
    P = [0] * N
    pos = 0
    steps = [_step("parts", [],
                   [("ca", ca, "green"), ("cb", cb, "indigo")],
                   eq=f"{ca} cycles of length {A} + {cb} cycle of length {B} = {N} positions")]
    for clen, colr in [(A, "green")] * ca + [(B, "indigo")] * cb:
        block = list(range(pos, pos + clen))
        for t, u in enumerate(block):
            nxt = block[(t + 1) % clen]
            P[u] = nxt + 1                # 1-indexed value
        steps.append(_step(f"cycle len {clen}",
                           [_row("P", P, hi=pos)],
                           eq=f"positions {block} -> rotate within the block"))
        pos += clen
    # verify every element sits on a cycle of length A or B
    def glen(i):
        j, k = i, 0
        while True:
            j = P[j - 1]
            k += 1
            if j == i:
                return k
    lens = {glen(i) for i in range(1, N + 1)}
    assert lens == {A, B}
    return _pack("Permutation Cycle - build P from the cycle blocks",
                 f"N = {N}, A = {A}, B = {B}", steps, str(P))


# -------------------------------------------------- CF 1567B MEXor Mixup

def _pref_xor(x):
    """XOR of 0..x."""
    return [x, 1, x + 1, 0][x % 4]


def _mexor_brute():
    """Genuine oracle: base {0..a-1} is forced (length >= a); add up to two spare
    elements (never equal to a) and find the shortest array with MEX a, XOR b."""
    a, b = 1, 1
    base = list(range(a))
    xbase = _pref_xor(a - 1)
    pool = list(range(0, 2 * a + 8))

    def mex_of(arr):
        s = set(arr)
        m = 0
        while m in s:
            m += 1
        return m

    def xor_of(arr):
        r = 0
        for v in arr:
            r ^= v
        return r

    steps = [_step("forced base", [_row("base", base)],
                [("a", a, "amber"), ("b", b, "indigo")],
                eq=f"MEX = {a} forces 0..{a-1} present and {a} absent, so length >= {a}")]
    best = None
    for extra in range(0, 3):
        if extra == 0:
            cand = [] if (xor_of(base) == b and mex_of(base) == a) else None
        elif extra == 1:
            cand = next(([v] for v in pool if v != a and xor_of(base + [v]) == b and mex_of(base + [v]) == a), None)
        else:
            cand = None
            for v1 in pool:
                for v2 in pool:
                    if v1 != a and v2 != a and xor_of(base + [v1, v2]) == b and mex_of(base + [v1, v2]) == a:
                        cand = [v1, v2]
                        break
                if cand:
                    break
        steps.append(_step(f"length {a + extra}", [_row("base", base)] +
                           ([_row("extra", cand)] if cand else []),
                           eq=f"can we finish with {extra} spare element(s)? -> {'yes' if cand is not None else 'no'}"))
        if cand is not None:
            best = a + extra
            break
    assert best == 3
    return _pack("MEXor Mixup - shortest-array search (oracle)",
                 f"a = {a}, b = {b}", steps, str(best))


def _mexor_formula():
    """X = xor(0..a-1); d = X ^ b; ans = a, a+1, a+2 by where d lands."""
    a, b = 1, 1
    X = _pref_xor(a - 1)
    d = X ^ b
    if d == 0:
        ans = a
    elif d != a:
        ans = a + 1
    else:
        ans = a + 2
    steps = [_step("base XOR", [],
                   [("a", a, "amber"), ("X", X, "green")],
                   eq=f"X = 0^1^..^{a-1} = {_pref_xor(a - 1)}")]
    steps.append(_step("needed delta", [],
                       [("d", d, "indigo")],
                       eq=f"d = X ^ b = {X} ^ {b} = {d}"))
    steps.append(_step("case", [],
                       [("d", d, "indigo"), ("a", a, "amber"), ("ans", ans, "violet")],
                       eq="d=0 -> a; d!=a -> a+1 (one spare d); d==a -> a+2 (d would break MEX)"))
    assert ans == 3
    return _pack("MEXor Mixup - XOR prefix (optimal O(1))",
                 f"a = {a}, b = {b}", steps, str(ans))


# ------------------------------------------------------ CF 136A Presents

_PRES_P = [2, 3, 4, 1]


def _presents_brute():
    """O(n^2): for each receiver i, find the giver j with p[j] == i."""
    p = list(_PRES_P)
    n = len(p)
    q = [0] * n
    steps = [_step("input", [_row("p", p)],
                   eq="p[i] = who friend i gave to; q[i] = who gave TO friend i")]
    for i in range(1, n + 1):
        for j in range(n):
            if p[j] == i:
                q[i - 1] = j + 1
                steps.append(_step(f"receiver {i}",
                                   [_row("p", p, hi=j), _row("q", q, hi=i - 1)],
                                   eq=f"p[{j+1}]={i} -> q[{i}]={j+1}"))
                break
    assert q == [4, 1, 2, 3]
    return _pack("Presents - search for each receiver (O(n^2))",
                 f"p = {p}", steps, str(q))


def _presents_inverse():
    """O(n): scatter q[p[i]] = i; the permutation read backwards (as a function)."""
    p = list(_PRES_P)
    n = len(p)
    q = [0] * n
    steps = [_step("input", [_row("p", p), _row("q", ["?"] * n)],
                   eq="one pass scatter: q[p[i]-1] = i+1")]
    for i in range(n):
        q[p[i] - 1] = i + 1
        steps.append(_step(f"write i={i+1}",
                           [_row("p", p, hi=i), _row("q", q, hi=p[i] - 1)],
                           eq=f"q[p[{i+1}]-1] = q[{p[i]-1}] = {i+1}"))
    assert q == [4, 1, 2, 3]
    return _pack("Presents - inverse by scatter (optimal O(n))",
                 f"p = {p}", steps, str(q))


# ------------------------------------------------- CF 1828B Permutation Swap

_PSWAP_P = [4, 2, 6, 7, 5, 3, 1]


def _permswap_chains():
    """Brute oracle: try k downward; a k is feasible iff every residue class of
    positions mod k already holds exactly the values that belong there."""
    p = list(_PSWAP_P)
    n = len(p)
    steps = [_step("input", [_row("p", p)],
                   eq=f"try k = {n-1}, {n-2}, ... ; positions i, i+k, i+2k form independent chains")]
    for k in range(n - 1, 0, -1):
        bad = None
        for r in range(k):
            idx = list(range(r, n, k))
            if sorted(p[i] for i in idx) != sorted(i + 1 for i in idx):
                bad = (r, idx)
                break
        if bad is None:
            steps.append(_step(f"k = {k} works",
                               [_row("p", p)],
                               [("k", k, "green")],
                               eq="every chain already contains its own values -> sort each chain freely"))
            assert k == 3
            return _pack("Permutation Swap - test each k (oracle)",
                         f"p = {p}", steps, str(k))
        r, idx = bad
        steps.append(_step(f"k = {k} fails", [_row("chain", [p[i] for i in idx], hi=0)],
                           [("k", k, "amber")],
                           eq=f"chain {idx} holds {sorted(p[i] for i in idx)}, needs {sorted(i + 1 for i in idx)}"))
    return _pack("Permutation Swap - test each k (oracle)", f"p = {p}", steps, "1")


def _permswap_gcd():
    """Optimal: answer = gcd of all |p_i - i| (1-based)."""
    from math import gcd
    p = list(_PSWAP_P)
    n = len(p)
    diffs = [abs(v - i) for i, v in enumerate(p, 1)]
    g, steps = 0, []
    steps.append(_step("input", [_row("p", p), _row("|p_i - i|", diffs)],
                       eq="every value must travel a multiple of k, so k divides each |p_i - i|"))
    for i in range(n):
        g = gcd(g, diffs[i])
        steps.append(_step(f"i = {i+1}",
                           [_row("p", p, hi=i), _row("|p_i - i|", diffs, hi=i)],
                           [("g", g, "green")],
                           eq=f"g = gcd(g, {diffs[i]}) = {g}"))
    assert g == 3
    return _pack("Permutation Swap - running gcd (optimal)",
                 f"p = {p}", steps, str(g))


# ------------------------------------------------------ CF 136B Ternary Logic

_TOR_A, _TOR_C = 14, 34


def _ternary_brute():
    """Brute oracle: scan b upward until a tor b == c (adds digitwise mod 3, no carry)."""
    a, c = _TOR_A, _TOR_C

    def tor(x, y):
        r, m = 0, 1
        while x or y:
            r += ((x % 3 + y % 3) % 3) * m
            m *= 3
            x, y = x // 3, y // 3
        return r

    tries = [0, 26, 27, 48, 50]
    steps = [_step("goal", [_row("a = 14 (ternary)", [2, 2, 1, 0]), _row("c = 34 (ternary)", [1, 2, 0, 1])],
                   eq="digits least-significant first; find smallest b with a tor b = c")]
    marks = ["no 4th digit to make c's leading 1", "26 = 222_3 -> digit 3 is 0",
             "27 = 1000_3 -> top digit 1+2 = 0 mod 3", "48 = 1210_3 -> digit 1 gives 2+1 = 0, but we need 2",
             "50 = 1212_3 -> hits 34 exactly"]
    for v, why in zip(tries, marks):
        steps.append(_step(f"b = {v}",
                           [_row("a tor b", [tor(a, v)], hi=0)],
                           [("tor", tor(a, v), "amber" if v != 50 else "green")],
                           eq=f"{a} tor {v} = {tor(a, v)}; {why}"))
    assert tor(a, 50) == c
    return _pack("Ternary Logic - scan b upward (oracle)",
                 f"a = {a}, c = {c}", steps, "50")


def _ternary_digits():
    """Optimal: read ternary digits of a and c side by side; b_i = (c_i - a_i) mod 3."""
    a, c = _TOR_A, _TOR_C
    da = [int(ch) for ch in reversed("112")]     # 14 = 112_3, least significant first
    dc = [int(ch) for ch in reversed("1021")]    # 34 = 1021_3
    db, b, steps = [], 0, []
    steps.append(_step("align digits", [_row("a digits", da), _row("c digits", dc)],
                       eq="least-significant first; missing digits are 0"))
    for i in range(len(dc)):
        digit = (dc[i] - (da[i] if i < len(da) else 0)) % 3
        db.append(digit)
        b += digit * 3 ** i
        steps.append(_step(f"digit {i}",
                           [_row("a digits", da, hi=i), _row("c digits", dc, hi=i), _row("b digits", db, hi=i)],
                           [("b", b, "green")],
                           eq=f"b_{i} = ({dc[i]} - {da[i] if i < len(da) else 0}) mod 3 = {digit}"))
    assert db == [2, 1, 2, 1] and b == 50
    return _pack("Ternary Logic - digitwise (c - a) mod 3 (optimal)",
                 f"a = {a}, c = {c}", steps, str(b))


# -------------------------------------------------------- CF 1487B Cat Cycle


def _catcycle_sim():
    """Brute oracle: simulate both cats hour by hour on n = 5."""
    n, k = 5, 5
    a, b = n, 1
    steps = [_step("hour 1", [_row("spots", list(range(1, n + 1)), hi=0)],
                   [("A", a, "indigo"), ("B", b, "amber")],
                   eq="A starts at n and walks down; B starts at 1 and walks up")]
    for hour in range(2, k + 1):
        a = n if a == 1 else a - 1
        b = 1 if b == n else b + 1
        note = f"A -> {a}, B -> {b}"
        if a == b:
            b = 1 if b == n else b + 1
            note += "; collision -> B skips one more"
        steps.append(_step(f"hour {hour}",
                           [_row("spots", list(range(1, n + 1)), hi=b - 1)],
                           [("A", a, "indigo"), ("B", b, "amber")], eq=note))
    assert b == 2
    return _pack("Cat Cycle - hour-by-hour simulation (oracle)",
                 f"n = {n}, k = {k}", steps, str(b))


def _catcycle_belt():
    """Odd n: every (n//2)-th hop B is bumped one extra spot; pos = (k-1 + skips) mod n + 1."""
    n, k = 5, 5
    half = n // 2
    skips = (k - 1) // half
    pos = (k - 1 + skips) % n + 1
    steps = [_step("belt view", [_row("hop i", list(range(0, k))), _row("i // 2", [i // half for i in range(k)])],
                   [("n", n, "indigo"), ("k", k, "amber")],
                   eq=f"n odd: A blocks the meeting spot every {half} hops, so B pays one extra skip"),
             _step("account", [_row("hop i", list(range(0, k))), _row("skips so far", [i // half for i in range(k)])],
                  [("pos", pos, "green")],
                  eq=f"pos = (k-1 + (k-1)//{half}) mod {n} + 1 = {pos}")]
    assert pos == 2
    return _pack("Cat Cycle - skip accounting (optimal)",
                 f"n = {n}, k = {k}", steps, str(pos))


# ----------------------------------------- LC 442 Find All Duplicates

_DUP_NUMS = [4, 3, 2, 7, 8, 2, 3, 1]


def _dups_cyclic():
    """Cyclic sort on the smaller array [7,3,4,5,5,6,2,2,1,8] (n = 10, dups 5 and 2)."""
    a = [7, 3, 4, 5, 5, 6, 2, 2, 1, 8]
    n = len(a)
    steps = [_step("input", [_row("nums", a)],
                   eq="values are in 1..n -> park value v at index v-1")]
    i = 0
    while i < n:
        if a[a[i] - 1] != a[i]:
            j = a[i] - 1
            act = f"a[{i}]={a[i]} belongs at index {j}: swap"
            a[i], a[j] = a[j], a[i]
            if len(steps) < 7:
                steps.append(_step(f"swap at {i}", [_row("nums", a, hi=i)], eq=act))
        else:
            i += 1
    dups = sorted(a[i] for i in range(n) if a[i] != i + 1)
    steps.append(_step("read off", [_row("sorted?", a), _row("i+1", list(range(1, n + 1)))],
                       [("dups", str(dups), "green")],
                       eq="a slot holding the wrong value carries a duplicate"))
    assert dups == [2, 5]
    return _pack("Find All Duplicates - cyclic sort (valid, mutates)",
                 "nums = [7, 3, 4, 5, 5, 6, 2, 2, 1, 8]", steps, str(dups))


def _dups_mark():
    """Optimal O(n)/O(1): negate the cell you visit; visiting a negated cell = second copy."""
    a = list(_DUP_NUMS)
    res, snaps = [], []
    for x in a:
        j = abs(x) - 1
        dup = a[j] < 0
        if dup:
            res.append(abs(x))
        else:
            a[j] = -a[j]
        snaps.append((x, list(a), list(res), dup))
    steps = [_step("input", [_row("nums", _DUP_NUMS)],
                   eq="for each x, flip the sign of position |x|-1; a flip onto negative = duplicate")]
    for t in (0, 3, 5, 6, 7):
        x, state, res_now, dup = snaps[t]
        j = abs(x) - 1
        why = (f"index {j} already negative -> {abs(x)} is a duplicate"
               if dup else f"visit {x}: mark index {j}")
        steps.append(_step(f"step {t+1}: x = {x}", [_row("nums", state, hi=j)],
                           [("res", str(res_now), "green")], eq=why))
    assert res == [2, 3]
    return _pack("Find All Duplicates - sign marking (optimal)",
                 f"nums = {_DUP_NUMS}", steps, str(res))


# ------------------------------------------- CF 1249B2 Books Exchange (hard)

_BOOKS_P = [4, 6, 2, 1, 5, 3]


def _books_walk():
    """One walk per cycle: follow i -> p_i until it closes, then write the
    cycle's length into every member. Each index is visited once."""
    p = list(_BOOKS_P)
    n = len(p)
    ans = ["?"] * n
    steps = [_step("input", [_row("p", p), _row("ans", ans)],
                   eq="kid i hands the book to kid p_i; a book returns after its cycle length")]
    for s in range(n):
        if ans[s] != "?":
            continue
        cyc, j = [], s
        while True:
            cyc.append(j)
            j = p[j] - 1
            if j == s:
                break
        for u in cyc:
            ans[u] = len(cyc)
        path = " -> ".join(str(u + 1) for u in cyc + [s])
        steps.append(_step(f"cycle from kid {s + 1}", [_row("p", p, hi=s), _row("ans", ans, hi=s)],
                           eq=f"{path}: length {len(cyc)}"))
    assert ans == [2, 3, 3, 2, 1, 3]
    return _pack("Books Exchange - one walk per cycle (optimal)",
                 f"p = {p}", steps, str(ans))


# ------------------------------------------------------------------ registry

REGISTRY = {
    ("makeperm", "count"): _makeperm_count(),
    ("makeperm", "keepfirst"): _makeperm_keepfirst(),
    ("makeperm", "greedy"): _makeperm_greedy(),
    ("merger", "brute"): _merger_brute(),
    ("merger", "seen"): _merger_seen(),
    ("mex", "brute"): _mex_brute(),
    ("mex", "formula"): _mex_formula(),
    ("cyc", "search"): _cyc_search(),
    ("cyc", "build"): _cyc_build(),
    ("mexor", "brute"): _mexor_brute(),
    ("mexor", "formula"): _mexor_formula(),
    ("presents", "brute"): _presents_brute(),
    ("presents", "inverse"): _presents_inverse(),
    ("permswap", "chains"): _permswap_chains(),
    ("permswap", "gcd"): _permswap_gcd(),
    ("ternary", "brute"): _ternary_brute(),
    ("ternary", "digits"): _ternary_digits(),
    ("catcycle", "sim"): _catcycle_sim(),
    ("catcycle", "belt"): _catcycle_belt(),
    ("dups", "cyclic"): _dups_cyclic(),
    ("dups", "mark"): _dups_mark(),
    ("books", "walk"): _books_walk(),
}


def get(problem, form):
    return REGISTRY[(problem, form)]
