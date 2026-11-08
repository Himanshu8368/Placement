"""Step frames for the hashing chapter.

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


def _mp(d):
    """A dict rendered the way the chapter prints it on a chip."""
    return "{" + ", ".join(f"{k}: {v}" for k, v in d.items()) + "}"


def _twosum_seen():
    """One-pass seen map on the chapter's sample."""
    a, target = [2, 7, 11, 15], 9
    seen, steps, final = {}, [], None
    for i, x in enumerate(a):
        want = target - x
        if want in seen:
            steps.append(chip_step(f"i = {i}: x = {x}, want {want}",
                                   [("i", i, "indigo"), ("x", x, "indigo"),
                                    ("want", want, "green"), ("seen", _mp(seen), "cyan")],
                                   eq=f"{want} seen at {seen[want]} -> return ({seen[want]}, {i})"))
            final = (seen[want], i)
            break
        steps.append(chip_step(f"i = {i}: x = {x}, want {want}",
                               [("i", i, "indigo"), ("x", x, "indigo"),
                                ("want", want, "amber"), ("seen", _mp(seen), "cyan")],
                               eq=f"{want} not seen -> store {x}: {i}"))
        seen[x] = i
    return _pack("Two Sum - query first, then insert",
                 "a = [2, 7, 11, 15], target = 9", steps, final)


def _ksum_count():
    """Greedy complement pairing with a count map."""
    a, k = [1, 2, 2, 3], 4
    cnt, pairs = {}, 0
    steps = [
        chip_step("start", [("k", k, "indigo"), ("pairs", pairs, "muted"),
                            ("counts", _mp(cnt), "cyan")], eq="")]
    for x in a:
        want = k - x
        if cnt.get(want, 0) > 0:
            cnt[want] -= 1
            pairs += 1
            steps.append(chip_step(f"x = {x}: want {want}",
                                   [("x", x, "indigo"), ("want", want, "green"),
                                    ("pairs", pairs, "green"), ("counts", _mp(cnt), "cyan")],
                                   eq=f"an open {want} pairs with this {x}"))
        else:
            cnt[x] = cnt.get(x, 0) + 1
            steps.append(chip_step(f"x = {x}: want {want}",
                                   [("x", x, "indigo"), ("want", want, "amber"),
                                    ("pairs", pairs, "muted"), ("counts", _mp(cnt), "cyan")],
                                   eq=f"no open {want} -> keep this {x}"))
    return _pack("Max Number of K-Sum Pairs - spend matches as they appear",
                 "a = [1, 2, 2, 3], k = 4", steps, pairs)


def _samediff_key():
    """Group by a[i] - i (1-based) and count pairs inside each group."""
    a = [1, 3, 3, 4]
    keys = [x - i for i, x in enumerate(a, start=1)]
    steps = [row_step("key every position by a[i] - i",
                      [("a", a, None), ("key", keys, None)],
                      eq="equal keys are exactly the wanted pairs")]
    total = 0
    for kv in sorted(set(keys)):
        where = [i for i, v in enumerate(keys) if v == kv]
        c = len(where)
        total += c * (c - 1) // 2
        steps.append(row_step(f"key {kv} appears at positions {[w + 1 for w in where]}",
                              [("a", a, None), ("key", keys, where[0])],
                              eq=f"C({c}, 2) = {c * (c - 1) // 2}"))
    return _pack("Same Differences - one key, then count within groups",
                 "a = [1, 3, 3, 4]  (CF sample, answer 3)", steps, total)


def _dupesii_last():
    """Last-seen map with a distance test."""
    a, k = [1, 2, 1], 2
    last, steps, final = {}, [], None
    for i, x in enumerate(a):
        if x in last:
            gap = i - last[x]
            hit = "True" if gap <= k else "no"
            steps.append(chip_step(f"i = {i}: x = {x} last seen at {last[x]}",
                                   [("i", i, "indigo"), ("x", x, "indigo"),
                                    ("gap", gap, "green" if gap <= k else "red"),
                                    ("last", _mp(last), "cyan")],
                                   eq=f"gap = {gap} {'<=' if gap <= k else '>'} k = {k} -> {hit}"))
            if gap <= k:
                final = True
                break
        else:
            steps.append(chip_step(f"i = {i}: x = {x} not seen yet",
                                   [("i", i, "indigo"), ("x", x, "indigo"),
                                    ("last", _mp(last), "cyan")],
                                   eq=f"store {x} -> {i}"))
        last[x] = i
    return _pack("Contains Duplicate II - remember only the newest index",
                 "a = [1, 2, 1], k = 2", steps, final)


def _majority_vote():
    """Boyer-Moore cancelling on the classic sample."""
    a = [2, 2, 1, 1, 1, 2, 2]
    cand, count, steps = None, 0, []
    for x in a:
        if count == 0:
            cand, count = x, 1
            note = f"pair-off left nothing -> new candidate {x}"
        elif x == cand:
            count += 1
            note = f"{x} matches -> +1"
        else:
            count -= 1
            note = f"{x} differs -> one {cand} and this {x} cancel"
        steps.append(chip_step(f"x = {x}",
                               [("candidate", cand, "indigo"), ("count", count,
                                "amber" if count > 0 else "red")],
                               eq=note))
    return _pack("Majority Element - Boyer-Moore voting",
                 "a = [2, 2, 1, 1, 1, 2, 2]  (majority guaranteed)", steps, cand)


def _majority2_vote():
    """Two-candidate voting, then a verification pass."""
    a = [1, 2, 2, 3, 1, 3, 1, 3]
    c1 = c2 = None
    n1 = n2 = 0
    steps = []
    for x in a:
        if c1 is not None and x == c1:
            n1 += 1
            note = f"{x} matches candidate 1"
        elif c2 is not None and x == c2:
            n2 += 1
            note = f"{x} matches candidate 2"
        elif n1 == 0:
            c1, n1 = x, 1
            note = f"slot 1 empty -> candidate 1 is {x}"
        elif n2 == 0:
            c2, n2 = x, 1
            note = f"slot 2 empty -> candidate 2 is {x}"
        else:
            n1 -= 1
            n2 -= 1
            note = f"{x} differs from both -> burn one of each"
        steps.append(chip_step(f"x = {x}",
                               [("c1", c1 if c1 is not None else "-", "indigo"),
                                ("n1", n1, "amber"),
                                ("c2", c2 if c2 is not None else "-", "violet"),
                                ("n2", n2, "pink")],
                               eq=note))
    from collections import Counter
    ans = sorted(v for v in (c1, c2) if v is not None and Counter(a)[v] > len(a) // 3)
    return _pack("Majority Element II - two counters, then verify",
                 "a = [1, 2, 2, 3, 1, 3, 1, 3]  (n/3 = 2)", steps, ans)


def _longest_starts():
    """Set membership: extend only from sequence starts."""
    a = [100, 4, 200, 1, 3, 2]
    s, steps, best = set(a), [], 0
    for x in a:
        if x - 1 in s:
            best_now = best
            steps.append(chip_step(f"x = {x}: {x - 1} is in the set",
                                   [("x", x, "muted"), ("start?", "no", "red"),
                                    ("best", best_now, "green")],
                                   eq="not a start - some earlier cell counts this run"))
        else:
            y = x + 1
            while y in s:
                y += 1
            best = max(best, y - x)
            steps.append(chip_step(f"x = {x}: {x - 1} is NOT in the set",
                                   [("x", x, "indigo"), ("start?", "yes", "green"),
                                    ("run", f"{x}..{y - 1}", "cyan"),
                                    ("best", best, "green")],
                                   eq=f"walk upward once -> length {y - x}"))
    return _pack("Longest Consecutive Sequence - start every walk at the bottom",
                 "a = [100, 4, 200, 1, 3, 2]", steps, best)


def _divp_rem():
    """Prefix remainders with a latest-index map (whole-array removal banned)."""
    a, p = [6, 3, 5, 2], 9
    rem = sum(a) % p
    last = {0: -1}
    pref, best, steps = 0, None, [
        chip_step("the remainder to remove",
                  [("total % p", rem, "amber"), ("last", _mp(last), "cyan"),
                   ("best", "-", "muted")],
                  eq=f"sum = {sum(a)}, need a proper subarray with sum % {p} == {rem}")]
    for i, x in enumerate(a):
        pref = (pref + x) % p
        want = (pref - rem) % p
        if want in last:
            ln = i - last[want]
            best = ln if best is None else min(best, ln)
            note = f"drop {last[want] + 1}..{i}: sum % {p} == {rem}, length {ln}"
            wc = "green"
        else:
            note = f"no earlier prefix is == {want} (mod {p})"
            wc = "amber"
        last[pref] = i
        steps.append(chip_step(f"i = {i}: x = {x}, prefix % {p} = {pref}",
                               [("pref", pref, "indigo"), ("want", want, wc),
                                ("last", _mp(last), "cyan"),
                                ("best", best if best is not None else "-", "muted")],
                               eq=note))
    return _pack("Make Sum Divisible by P - latest index per remainder",
                 "a = [6, 3, 5, 2], p = 9  (LC sample)", steps, best)


_ALL = None


def get(problem, form):
    """Return ``(title, input line, steps, final)`` for one solution form."""
    global _ALL
    if _ALL is None:
        _ALL = {
            ("twosum", "seen"): _twosum_seen(),
            ("ksum", "count"): _ksum_count(),
            ("samediff", "key"): _samediff_key(),
            ("dupesii", "last"): _dupesii_last(),
            ("majority", "vote"): _majority_vote(),
            ("majority2", "vote2"): _majority2_vote(),
            ("longest", "starts"): _longest_starts(),
            ("divp", "rem"): _divp_rem(),
        }
    return _ALL[(problem, form)]
