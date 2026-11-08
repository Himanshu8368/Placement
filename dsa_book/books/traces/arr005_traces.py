"""Step frames for the intro-arrays chapter.

Drawing lives in ``dsa_viz.story``. ``get`` returns data only.
Each story walks the chapter's own sample, at most eight frames, and the
numbers come from running the algorithm - never from a hand-typed table.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import chip_step, row_step, show


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    if not 1 <= len(steps) <= 8:
        raise AssertionError(f"{title}: {len(steps)} frames")
    return (title, inp, steps, text)


def _dash(v):
    return "-" if v is None else v


def _third_slide():
    a = [2, 2, 3, 1]
    first = second = third = None
    steps = [chip_step("three slots, all empty",
                       [("1st", _dash(first), "muted"), ("2nd", _dash(second), "muted"),
                        ("3rd", _dash(third), "muted")],
                       "keep first > second > third, '-' = unfilled")]
    for x in a:
        if x in (first, second, third):
            steps.append(chip_step(f"read {x}", [("1st", _dash(first), "green"),
                                                 ("2nd", _dash(second), "green"),
                                                 ("3rd", _dash(third), "green")],
                                   f"{x} already a keeper -> skip"))
            continue
        if first is None or x > first:
            third, second, first = second, first, x
            eq = f"{x} > 1st -> slide all three down"
        elif second is None or x > second:
            third, second = second, x
            eq = f"{x} > 2nd -> slide into 2nd"
        elif third is None or x > third:
            third = x
            eq = f"{x} fills 3rd"
        else:
            eq = f"{x} below 3rd -> ignore"
        steps.append(chip_step(f"read {x}", [("1st", _dash(first), "indigo"),
                                             ("2nd", _dash(second), "indigo"),
                                             ("3rd", _dash(third), "indigo")], eq))
    ans = first if third is None else third
    return _pack("Third Maximum Number - slide three slots", "a = [2, 2, 3, 1]", steps, ans)


def _movez_compact():
    a = [0, 1, 0, 3, 12]
    b = list(a)
    w = 0
    steps = [row_step("compact the non-zeros to the front", [("a", b[:], None)],
                      f"w = {w}")]
    for i in range(len(a)):
        x = b[i]
        if x != 0:
            b[w] = x
            w += 1
            eq = f"copy {x} -> a[{w - 1}], w = {w}"
        else:
            eq = f"{x}: leave for the tail fill"
        steps.append(row_step(f"read a[{i}] = {x}", [("a", b[:], i)], eq))
    for i in range(w, len(a)):
        b[i] = 0
    steps.append(row_step("paint the tail with zeros", [("a", b[:], None)],
                          f"a[{w}:] = 0"))
    return _pack("Move Zeroes - read/write compaction", "a = [0, 1, 0, 3, 12]", steps, b)


def _dupes_twice():
    a = [1, 1, 1, 2, 2, 3]
    b = list(a)
    w = 0
    steps = [row_step("sorted; keep at most two of each", [("a", b[:], None)],
                      f"w = {w}")]
    for i in range(len(a)):
        x = b[i]
        if w < 2 or x != b[w - 2]:
            b[w] = x
            w += 1
            eq = f"keep {x} -> a[{w - 1}], w = {w}"
        else:
            eq = f"drop {x}: equals a[{w - 2}] (would be a third)"
        steps.append(row_step(f"read a[{i}] = {x}", [("a", b[:], i)], eq))
    return _pack("Remove Duplicates II - a two-back look", "a = [1, 1, 1, 2, 2, 3]",
                 steps, b[:w])


def _rotate_three():
    a = [1, 2, 3, 4, 5, 6, 7]
    n, k = len(a), 3

    def rev(arr, l, r):
        r -= 1
        while l < r:
            arr[l], arr[r] = arr[r], arr[l]
            l += 1
            r -= 1

    b = list(a)
    steps = [row_step("rotate right by k = 3", [("a", b[:], None)],
                      "reverse all, then reverse each part")]
    rev(b, 0, n)
    steps.append(row_step("reverse [0, n)", [("a", b[:], None)],
                          "the two blocks swap order, both upside-down"))
    rev(b, 0, k)
    steps.append(row_step("reverse [0, k)", [("a", b[:], None)], "front block righted"))
    rev(b, k, n)
    steps.append(row_step("reverse [k, n)", [("a", b[:], None)], "back block righted"))
    return _pack("Rotate Array - three reversals", "a = [1..7], k = 3", steps, b)


def _plusone_carry():
    a = [9, 9, 9]
    b = list(a)
    i = len(b) - 1
    carry = 1
    steps = [row_step("add one, starting at the last digit", [("a", b[:], i)],
                      "carry = 1")]
    while i >= 0 and carry:
        s = b[i] + carry
        b[i] = s % 10
        carry = s // 10
        steps.append(row_step(f"digit {i}: {a[i]} + carry", [("a", b[:], i)],
                              f"{s} -> write {s % 10}, carry {carry}"))
        i -= 1
    if carry:
        b = [1] + b
        steps.append(row_step("a carry leaves the front", [("a", b[:], 0)],
                              "all nines -> a new leading 1"))
    return _pack("Plus One - propagate the carry", "digits = [9, 9, 9]", steps, b)


def _build_encode():
    a = [0, 2, 1, 5, 3, 4]
    n = len(a)
    b = list(a)
    steps = [row_step("store old + n * new in every cell", [("a", b[:], None)], f"n = {n}")]
    for i in range(n):
        new = b[b[i]] % n                       # b[i] is still the original value here
        b[i] += n * new
        steps.append(row_step(f"i = {i}: new = a[{b[i] % n}] % {n} = {new}", [("a", b[:], i)],
                              f"a[{i}] = {b[i] % n} + {n}*{new} = {b[i]}"))
    out = [x // n for x in b]
    steps.append(row_step("read the new values: a[i] // n", [("a", out[:], None)], "answer"))
    if len(steps) > 8:                          # keep the board short: drop the middle rows
        steps = steps[:3] + steps[-5:]
    return _pack("Build Array - two values per cell", "nums = [0, 2, 1, 5, 3, 4]", steps, out)


def _smaller_freq():
    nums = [8, 1, 2, 2, 3]
    freq = [0] * 9
    for x in nums:
        freq[x] += 1
    steps = [row_step("count each value (cells 0..8)", [("cnt", freq[:], None)], "one pass over nums")]
    less, run = [0] * 9, 0
    for v in range(9):
        less[v] = run                           # everything counted so far is < v
        run += freq[v]
        if v in (1, 2, 3, 8):
            steps.append(row_step(f"v = {v}: less[{v}] = {less[v]}", [("cnt", freq[:], v), ("less", less[:], v)],
                                  f"then run += cnt[{v}] -> {run}"))
    out = [less[x] for x in nums]
    steps.append(row_step("answer = less[nums[i]]", [("less", less[:], None)], str(out)))
    return _pack("How Many Smaller - counts, then a running sum", "nums = [8, 1, 2, 2, 3]", steps, out)


_ALL = None


def get(problem, form):
    """Return ``(title, input line, steps, final)`` for one solution form."""
    global _ALL
    if _ALL is None:
        _ALL = {
            ("third", "slide"): _third_slide(),
            ("movez", "compact"): _movez_compact(),
            ("dupes", "twice"): _dupes_twice(),
            ("rotate", "three"): _rotate_three(),
            ("plusone", "carry"): _plusone_carry(),
            ("build", "encode"): _build_encode(),
            ("smaller", "freq"): _smaller_freq(),
        }
    return _ALL[(problem, form)]
