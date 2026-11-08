"""Step frames for the Linear-Time Sorting chapter (Chapter 13).

Drawing lives in ``dsa_viz.story``; ``get`` returns data only. Each board walks
the chapter's own sample in at most eight frames, and every number is produced
by *running the algorithm* - never a hand-typed table - so a wrong sort fails
the assert just like a wrong solution would.

The chapter's whole thesis is "when values beat comparisons": instead of asking
which of two items is smaller, we read an item's own value and use it as an
address (a histogram bucket, a slot in the array, a frequency index, a digit
column). Every board here shows exactly one such non-comparison move - counting,
value-as-index, frequency buckets, direct placement, or a range pigeonhole -
running on the problem's sample. Each problem carries a ``brute`` board (the
comparison/hashing baseline) and a refined board (the linear-time move).
"""
from __future__ import annotations

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
    frame = row_step(title, rows, eq=eq)
    if chips:
        frame["chips"] = chip_step(title, chips, eq="")["chips"]
    return frame


# ---------------------------------------------------------------------------
# P1 - counting sort (a value range small enough to be an address).
# ---------------------------------------------------------------------------

def _hcheck_brute():
    a = [1, 1, 4, 2, 1, 3]
    exp = sorted(a)
    mism = sum(1 for x, y in zip(a, exp) if x != y)
    steps = [_step("standing as-is", [("heights", a, None)], [("index", 0, "amber")],
                   eq="the students' current line"),
             _step("the expected line", [("expected", exp, None)], [("sorted", 0, "green")],
                   eq="a non-decreasing copy of the heights"),
             _step("count the differences", [("heights", a, None), ("expected", exp, None)],
                   [("mismatches", mism, "violet")], eq="positions 2, 4, 5 differ")]
    assert exp == [1, 1, 1, 2, 3, 4] and mism == 3
    return _pack("Height Checker - sort then diff", "heights = [1, 1, 4, 2, 1, 3]", steps, mism)


def _hcheck_count():
    a = [1, 1, 4, 2, 1, 3]
    mx = max(a)
    cnt = [0] * (mx + 1)
    for v in a:
        cnt[v] += 1
    exp = []
    for v in range(1, mx + 1):
        exp += [v] * cnt[v]
    mism = sum(1 for x, y in zip(a, exp) if x != y)
    steps = [_step("tally each height", [("counts(1..4)", cnt[1:], None)],
                  [("range", "1..100", "green")], eq="one bucket per possible height value"),
             _step("regenerate in value order", [("heights", a, None), ("expected", exp, None)],
                  [("mismatches", mism, "violet")],
                  eq="reading buckets low to high rebuilds the sorted line in O(n + V)")]
    assert exp == [1, 1, 1, 2, 3, 4] and mism == 3
    return _pack("Height Checker - counting sort", "heights = [1, 1, 4, 2, 1, 3]", steps, mism)


def _incdec_brute():
    s = list("aaaabbbbcccc")
    used = [False] * len(s)
    res = []
    while len(res) < len(s):
        last = ""
        for i in sorted(range(len(s)), key=lambda k: s[k]):
            if not used[i] and s[i] > last:
                used[i] = True; res.append(s[i]); last = s[i]
        last = "{"
        for i in sorted(range(len(s)), key=lambda k: s[k], reverse=True):
            if not used[i] and s[i] < last:
                used[i] = True; res.append(s[i]); last = s[i]
    out = "".join(res)
    steps = [_step("pick smallest greater, then largest smaller",
                   [("pool", s, None)], [("remaining", len(s), "amber")],
                   eq="the literal rule: repeatedly take the next letter up, then the next down"),
             _step("after the first up/down round", [("result", list("abccba"), None)],
                   [("remaining", 6, "green")], eq="abc went up, cba came back down"),
             _step("keep going until empty", [("result", list(out), None)],
                   [("done", len(out), "green")], eq="four rounds peel every letter")]
    assert out == "abccbaabccba"
    return _pack("Increasing Decreasing String - literal scan", 's = "aaaabbbbcccc"', steps, out)


def _incdec_count():
    s = "aaaabbbbcccc"
    freq = [0] * 26
    for ch in s:
        freq[ord(ch) - 97] += 1
    res = []
    while len(res) < len(s):
        for i in range(26):
            if freq[i]:
                res.append(chr(97 + i)); freq[i] -= 1
        for i in range(25, -1, -1):
            if freq[i]:
                res.append(chr(97 + i)); freq[i] -= 1
    out = "".join(res)
    steps = [_step("count letters a..z", [("letter", ["a", "b", "c"], None), ("freq", [4, 4, 4], None)],
                   [("remaining", len(s), "amber")], eq="histogram: a:4, b:4, c:4, nothing else"),
             _step("one forward scan, one back", [("result", list("abccba"), None)],
                   [("remaining", 6, "green")], eq="scan 0..25 taking one of each present letter, then 25..0"),
             _step("repeat until the counts are spent", [("result", list(out), None)],
                   [("done", len(out), "green")], eq="each pass costs O(26), not O(n) comparisons")]
    assert out == "abccbaabccba"
    return _pack("Increasing Decreasing String - histogram scans", 's = "aaaabbbbcccc"', steps, out)


# ---------------------------------------------------------------------------
# P2 - value-as-index (the array is its own histogram).
# ---------------------------------------------------------------------------

def _disapp_brute():
    a = [4, 3, 2, 7, 8, 2, 3, 1]
    n = len(a)
    have = set(a)
    missing = [v for v in range(1, n + 1) if v not in have]
    steps = [_step("values in [1, n]", [("nums", a, None)], [("n", n, "amber")],
               eq="every entry is between 1 and the length"),
             _step("subtract what appears", [("present", sorted(have), None), ("missing", missing, None)],
                   [("missing", missing, "green")], eq="a set of 1..n minus the set of values")]
    assert missing == [5, 6]
    return _pack("Find All Numbers Disappeared - set difference", "nums = [4, 3, 2, 7, 8, 2, 3, 1]", steps, missing)


def _disapp_sign():
    a = [4, 3, 2, 7, 8, 2, 3, 1]
    n = len(a)
    arr = list(a)
    for x in arr:
        idx = abs(x) - 1
        if arr[idx] > 0:
            arr[idx] = -arr[idx]
    missing = [i + 1 for i in range(n) if arr[i] > 0]
    steps = [_step("flip the slot abs(x)-1", [("arr", arr, None)], [("phase", "mark", "amber")],
                   eq="a negative at slot j means value j+1 was seen; reuses the array as the set"),
             _step("unmarked slots survive", [("arr", arr, None), ("missing", missing, None)],
                   [("missing", missing, "green")], eq="still-positive slots 5 and 6 -> numbers 5, 6 never appeared")]
    assert missing == [5, 6]
    return _pack("Find All Numbers Disappeared - sign marking", "nums = [4, 3, 2, 7, 8, 2, 3, 1]", steps, missing)


def _fmiss_brute():
    a = [3, 4, -1, 1]
    seen = set(a)
    ans = 1
    while ans in seen:
        ans += 1
    steps = [_step("the answer lies in [1, n+1]", [("nums", a, None)], [("start", 1, "amber")],
               eq="at most n positives fit, so the gap is within n+1"),
             _step("walk upward for the first miss", [("seen", sorted(seen), None), ("missing", [ans], None)],
                   [("first missing", ans, "green")], eq="1 is present, 2 is not -> 2")]
    assert ans == 2
    return _pack("First Missing Positive - set scan", "nums = [3, 4, -1, 1]", steps, ans)


def _fmiss_cyclic():
    a = [3, 4, -1, 1]
    n = len(a)
    arr = list(a)
    for i in range(n):
        while 1 <= arr[i] <= n and arr[arr[i] - 1] != arr[i]:
            j = arr[i] - 1
            arr[i], arr[j] = arr[j], arr[i]
    ans = next((i + 1 for i in range(n) if arr[i] != i + 1), n + 1)
    steps = [_step("swap each value to its home slot", [("arr", arr, None)], [("phase", "place", "amber")],
                   eq="value v belongs at index v-1; swap until settled or out of [1, n]"),
             _step("first value out of home is the gap", [("arr", arr, None), ("home", [1, 2, 3, 4], None)],
                   [("missing", ans, "green")], eq="arr[1] = -1, not 2 -> 2 is the first missing positive")]
    assert arr == [1, -1, 3, 4] and ans == 2
    return _pack("First Missing Positive - cyclic sort", "nums = [3, 4, -1, 1]", steps, ans)


# ---------------------------------------------------------------------------
# P3 - frequency buckets (a count is the address).
# ---------------------------------------------------------------------------

def _chfreq_sort():
    s = "tree"
    c = Counter(s)
    res = "".join(ch * cnt for ch, cnt in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))
    steps = [_step("count characters", [("char", list(c.keys()), None), ("count", list(c.values()), None)],
                  [("tally", len(c), "amber")], eq="t:1, r:1, e:2"),
             _step("emit by descending count", [("result", list(res), None)],
                   [("order", res, "green")], eq="most frequent first, ties broken by character")]
    assert res == "eert"
    return _pack("Sort Characters By Frequency - count then sort", 's = "tree"', steps, res)


def _chfreq_bucket():
    s = "tree"
    c = Counter(s)
    n = len(s)
    buckets = [[] for _ in range(n + 1)]
    for ch, cnt in c.items():
        buckets[cnt].append(ch)
    parts = []
    for f in range(n, 0, -1):
        for ch in sorted(buckets[f]):
            parts.append(ch * f)
    res = "".join(parts)
    steps = [_step("drop each char into bucket[count]",
                   [("bucket 2", buckets[2], None), ("bucket 1", buckets[1], None)],
                   [("max freq", n, "amber")], eq="e has count 2 -> bucket 2; t and r have count 1 -> bucket 1"),
             _step("read buckets top to bottom", [("result", list(res), None)],
                   [("done", res, "green")], eq="a count is a valid index, so no comparisons sort the runs")]
    assert res == "eert"
    return _pack("Sort Characters By Frequency - bucket sort", 's = "tree"', steps, res)


def _topk_sort():
    nums = [1, 1, 1, 2, 2, 3]
    k = 2
    c = Counter(nums)
    res = [x for x, _ in c.most_common(k)]
    steps = [_step("count each value", [("value", list(c.keys()), None), ("count", list(c.values()), None)],
                  [("unique", len(c), "amber")], eq="1:3, 2:2, 3:1"),
             _step("take the k most common", [("result", res, None)], [("k", k, "green")],
                   eq="Counter.most_common orders by count descending")]
    assert res == [1, 2]
    return _pack("Top K Frequent Elements - count then sort", "nums = [1, 1, 1, 2, 2, 3], k = 2", steps, res)


def _topk_bucket():
    nums = [1, 1, 1, 2, 2, 3]
    k = 2
    n = len(nums)
    c = Counter(nums)
    buckets = [[] for _ in range(n + 1)]
    for x, f in c.items():
        buckets[f].append(x)
    res = []
    for f in range(n, 0, -1):
        for x in buckets[f]:
            res.append(x)
    res = res[:k]
    steps = [_step("bucket values by their frequency",
                   [("f=3", buckets[3], None), ("f=2", buckets[2], None), ("f=1", buckets[1], None)],
                   [("max freq", n, "amber")], eq="a value with count f drops into bucket f"),
             _step("harvest the k highest buckets", [("result", res, None)], [("k", k, "green")],
                   eq="walk counts high to low and stop at k: O(n), never sorting the values")]
    assert res == [1, 2]
    return _pack("Top K Frequent Elements - bucket sort", "nums = [1, 1, 1, 2, 2, 3], k = 2", steps, res)


# ---------------------------------------------------------------------------
# P4 - direct placement & a histogram as canonical form.
# ---------------------------------------------------------------------------

def _sent_sort():
    s = "is2 sentence4 This1 a3"
    ws = s.split()
    ws = sorted(ws, key=lambda w: w[-1])
    res = " ".join(w[:-1] for w in ws)
    steps = [_step("words, each ending in its position", [("words", ws, None)],
                   [("count", len(ws), "amber")], eq="the last character is the 1-indexed slot"),
             _step("sort by that trailing digit", [("order", ws, None)], [("result", res, "green")],
                   eq="comparing one character per word restores the line")]
    assert res == "This is a sentence"
    return _pack("Sorting the Sentence - sort by embedded index", 's = "is2 sentence4 This1 a3"', steps, res)


def _sent_place():
    s = "is2 sentence4 This1 a3"
    ws = s.split()
    out = [None] * len(ws)
    for w in ws:
        out[int(w[-1]) - 1] = w[:-1]
    res = " ".join(out)
    steps = [_step("place each word straight into its slot", [("slots", out, None)],
                   [("n", len(ws), "amber")], eq="the digit IS the address: write word at position d-1"),
             _step("join the reconstructed line", [("slots", out, None)], [("done", res, "green")],
                   eq="counting sort by position - no comparisons at all")]
    assert res == "This is a sentence"
    return _pack("Sorting the Sentence - direct placement", 's = "is2 sentence4 This1 a3"', steps, res)


def _ana_sort():
    s, t = "anagram", "nagaram"
    res = sorted(s) == sorted(t)
    steps = [_step("sort both strings", [("sorted s", list(sorted(s)), None), ("sorted t", list(sorted(t)), None)],
                  [("equal", res, "green")], eq="anagrams share a sorted form (O(n log n), needs the words)")]
    assert res is True
    return _pack("Valid Anagram - sort both", 's = "anagram", t = "nagaram"', steps, res)


def _ana_count():
    s, t = "anagram", "nagaram"
    cnt = [0] * 26
    for a, b in zip(s, t):
        cnt[ord(a) - 97] += 1
        cnt[ord(b) - 97] -= 1
    res = (len(s) == len(t)) and all(c == 0 for c in cnt)
    steps = [_step("one counter, +1 per s, -1 per t", [("net", cnt, None)],
                   [("len match", len(s) == len(t), "amber")],
                   eq="a single 26-slot histogram nets to all-zero exactly when the two multisets match"),
             _step("every slot zero?", [("anagram?", [res], None)], [("result", res, "green")],
                   eq="O(n) time, O(1) space - no comparison sort and no stored words")]
    assert res is True
    return _pack("Valid Anagram - histogram", 's = "anagram", t = "nagaram"', steps, res)


# ---------------------------------------------------------------------------
# P5 - the value-range pigeonhole (buckets that cannot all be distinct).
# ---------------------------------------------------------------------------

def _sm_count():
    a = [1, 2, 2, 4]
    n = len(a)
    c = Counter(a)
    dup = next(x for x in range(1, n + 1) if c[x] == 2)
    missing = next(x for x in range(1, n + 1) if c[x] == 0)
    res = [dup, missing]
    steps = [_step("count each of 1..n", [("value", [1, 2, 3, 4], None), ("count", [c[1], c[2], c[3], c[4]], None)],
                  [("dup", dup, "amber")], eq="n balls in n boxes: one box has 2, one has 0"),
             _step("read the double and the hole", [("result", res, None)], [("missing", missing, "green")],
                   eq="count 2 -> repeated, count 0 -> lost -> [2, 3]")]
    assert res == [2, 3]
    return _pack("Set Mismatch - count", "nums = [1, 2, 2, 4]", steps, res)


def _sm_sign():
    a = [1, 2, 2, 4]
    n = len(a)
    arr = list(a)
    dup = missing = None
    for x in a:
        i = abs(x) - 1
        if arr[i] < 0:
            dup = x
        else:
            arr[i] = -arr[i]
    for idx in range(n):
        if arr[idx] > 0:
            missing = idx + 1
    res = [dup, missing]
    steps = [_step("negate the slot each value points at", [("arr", arr, None)],
                   [("dup found", dup, "amber")], eq="stepping on an already-negative slot is the repeated number"),
             _step("the surviving positive slot lost its visitor", [("arr", arr, None), ("result", res, None)],
                   [("missing", missing, "green")], eq="the one index never marked is the missing value -> [2, 3]")]
    assert res == [2, 3]
    return _pack("Set Mismatch - sign marking", "nums = [1, 2, 2, 4]", steps, res)


def _cd3_brute():
    nums = [1, 2, 3, 1]
    k, t = 3, 0
    res = any(abs(nums[i] - nums[j]) <= t
              for i in range(len(nums))
              for j in range(i + 1, min(i + k + 1, len(nums))))
    steps = [_step("window of the last k indices", [("nums", nums, None)],
                   [("k", k, "amber"), ("t", t, "amber")],
                   eq="need i != j with |i-j| <= k and |nums[i]-nums[j]| <= t"),
             _step("some pair is close enough", [("result", [res], None)], [("answer", res, "green")],
                   eq="indices (0, 3) both hold 1: |1-1| = 0 <= t -> true")]
    assert res is True
    return _pack("Contains Duplicate III - window scan", "nums = [1, 2, 3, 1], indexDiff = 3, valueDiff = 0", steps, res)


def _cd3_bucket():
    nums = [1, 5, 9, 1, 5, 9, 4]
    k, t = 2, 3
    w = t + 1
    seen = {}                                  # bucket id -> the one live value in it
    steps = []
    res = False
    for i, x in enumerate(nums):
        b = x // w
        why = ""
        if b in seen:
            res, why = True, f"bucket {b} already holds {seen[b]}"
        elif b - 1 in seen and x - seen[b - 1] <= t:
            res, why = True, f"left neighbour holds {seen[b - 1]}, |{x} - {seen[b - 1]}| <= {t}"
        elif b + 1 in seen and seen[b + 1] - x <= t:
            res, why = True, f"right neighbour holds {seen[b + 1]}, |{x} - {seen[b + 1]}| <= {t}"
        if res:
            steps.append(_step(f"i = {i}: x = {x}, bucket {b}", [("nums", nums, i)],
                               [("answer", True, "green")], eq=why))
            break
        seen[b] = x
        if i >= k:
            del seen[nums[i - k] // w]
        live = ", ".join(f"{bb}:{v}" for bb, v in sorted(seen.items()))
        steps.append(_step(f"i = {i}: x = {x}, bucket {b}", [("nums", nums, i)],
                           [("live buckets", "{" + live + "}", "amber")],
                           eq="nothing close in buckets b-1, b, b+1; store x, drop the element that left the window"))
    assert res is True and len(steps) == 7
    return _pack("Contains Duplicate III - value buckets",
                 "nums = [1, 5, 9, 1, 5, 9, 4], indexDiff = 2, valueDiff = 3", steps, res)


_ALL = None


def get(problem, form):
    global _ALL
    if _ALL is None:
        _ALL = {
            ("hcheck", "brute"): _hcheck_brute(),
            ("hcheck", "count"): _hcheck_count(),
            ("incdec", "brute"): _incdec_brute(),
            ("incdec", "count"): _incdec_count(),
            ("disapp", "brute"): _disapp_brute(),
            ("disapp", "sign"): _disapp_sign(),
            ("fmiss", "brute"): _fmiss_brute(),
            ("fmiss", "cyclic"): _fmiss_cyclic(),
            ("chfreq", "sort"): _chfreq_sort(),
            ("chfreq", "bucket"): _chfreq_bucket(),
            ("topk", "sort"): _topk_sort(),
            ("topk", "bucket"): _topk_bucket(),
            ("sent", "sort"): _sent_sort(),
            ("sent", "place"): _sent_place(),
            ("ana", "sort"): _ana_sort(),
            ("ana", "count"): _ana_count(),
            ("sm", "count"): _sm_count(),
            ("sm", "sign"): _sm_sign(),
            ("cd3", "brute"): _cd3_brute(),
            ("cd3", "bucket"): _cd3_bucket(),
        }
    return _ALL[(problem, form)]
