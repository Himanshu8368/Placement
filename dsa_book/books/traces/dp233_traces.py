"""Step boards for the string DP chapter (two sequences at once)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import chip_step, grid_step, line_step, row_step, run_call, show

MOD = 10**9 + 7


def _pack(title, inp, steps, answer):
    if isinstance(answer, bool):
        text = "true" if answer else "false"
    else:
        text = answer if isinstance(answer, str) else show(answer)
    steps = list(steps)
    if len(steps) > 8:
        steps = steps[:7] + steps[-1:]
    return (title, inp, steps, text)


def lcs_table(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp


def lcs_len(a, b):
    return lcs_table(a, b)[-1][-1]


def ascii_table(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + ord(a[i - 1])
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + ord(b[j - 1])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + ord(a[i - 1]), dp[i][j - 1] + ord(b[j - 1]))
    return dp


def edit_table(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp


def scs_string(a, b):
    dp = lcs_table(a, b)
    i, j = len(a), len(b)
    out = []
    while i and j:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            out.append(a[i - 1])
            i -= 1
        else:
            out.append(b[j - 1])
            j -= 1
    out.extend(reversed(a[:i]))
    out.extend(reversed(b[:j]))
    out.reverse()
    return "".join(out)


def dist_table(s, t):
    n, m = len(s), len(t)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = 1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i - 1][j]
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]
    return dp


def inter_table(a, b, c):
    if len(a) + len(b) != len(c):
        return [[False]]
    n, m = len(a), len(b)
    dp = [[False] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = True
    for i in range(n + 1):
        for j in range(m + 1):
            if i and dp[i - 1][j] and a[i - 1] == c[i + j - 1]:
                dp[i][j] = True
            if j and dp[i][j - 1] and b[j - 1] == c[i + j - 1]:
                dp[i][j] = True
    return dp


def wild_table(s, p):
    n, m = len(s), len(p)
    dp = [[False] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = True
    for j in range(1, m + 1):
        if p[j - 1] == "*":
            dp[0][j] = dp[0][j - 1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if p[j - 1] == "*":
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
            elif p[j - 1] == "?" or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    return dp


def regex_table(s, p):
    n, m = len(s), len(p)
    dp = [[False] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = True
    for i in range(n + 1):
        for j in range(1, m + 1):
            if p[j - 1] == "*":
                dp[i][j] = dp[i][j - 2] if j >= 2 else False
                if i and j >= 2 and (p[j - 2] == "." or p[j - 2] == s[i - 1]):
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
            elif i and (p[j - 1] == "." or p[j - 1] == s[i - 1]):
                dp[i][j] = dp[i - 1][j - 1]
    return dp


def _pi(s):
    pi = [0] * len(s)
    k = 0
    for i in range(1, len(s)):
        while k and s[i] != s[k]:
            k = pi[k - 1]
        if s[i] == s[k]:
            k += 1
        pi[i] = k
    return pi


def bracket_count(n, s):
    m = len(s)
    pi = _pi(s)
    go = [[0, 0] for _ in range(m + 1)]
    for st in range(m):
        for bi, ch in enumerate("()"):
            k = st
            while k and (k == m or s[k] != ch):
                k = pi[k - 1] if k < m else 0
            if k < m and s[k] == ch:
                k += 1
            go[st][bi] = k
    go[m][0] = go[m][1] = m
    dp = [[[0] * (m + 1) for _ in range(n + 1)] for _ in range(2 * n + 1)]
    dp[0][0][0] = 1
    for i in range(2 * n):
        for b in range(n + 1):
            for k in range(m + 1):
                ways = dp[i][b][k]
                if not ways:
                    continue
                if b + 1 <= n:
                    nk = go[k][0]
                    dp[i + 1][b + 1][nk] = (dp[i + 1][b + 1][nk] + ways) % MOD
                if b:
                    nk = go[k][1]
                    dp[i + 1][b - 1][nk] = (dp[i + 1][b - 1][nk] + ways) % MOD
    return dp[2 * n][0][m]


def _evil_go(evil):
    m = len(evil)
    pi = _pi(evil)
    go = [[0] * 26 for _ in range(m + 1)]
    for st in range(m):
        for c in range(26):
            ch = chr(c + 97)
            k = st
            while k and evil[k] != ch:
                k = pi[k - 1]
            if evil[k] == ch:
                k += 1
            go[st][c] = k
    return go


def good_count(n, s1, s2, evil):
    go = _evil_go(evil)
    m = len(evil)
    from functools import lru_cache

    @lru_cache(None)
    def f(i, k, lo, hi):
        if k == m:
            return 0
        if i == n:
            return 1
        left = ord(s1[i]) - 97 if lo else 0
        right = ord(s2[i]) - 97 if hi else 25
        ans = 0
        for c in range(left, right + 1):
            nk = go[k][c]
            if nk == m:
                continue
            ans = (ans + f(i + 1, nk, lo and c == left, hi and c == right)) % MOD
        return ans

    return f(0, 0, 1, 1)


def _grid_bits(grid):
    return [["T" if c else "F" for c in row] for row in grid]


def _lcs(kind):
    a, b = "abcde", "ace"
    dp = lcs_table(a, b)
    assert dp[-1][-1] == 3
    if kind == "rec":
        def build(log):
            def f(i, j):
                log.call(f"f({i},{j})")
                if i == 0 or j == 0:
                    return log.ret(0)
                if a[i - 1] == b[j - 1]:
                    return log.ret(f(i - 1, j - 1) + 1)
                return log.ret(max(f(i - 1, j), f(i, j - 1)))
            return f(len(a), len(b))
        answer, steps = run_call(build)
        return _pack("LCS — recursion", "abcde / ace", steps, answer)
    steps = [
        grid_step("prefixes", dp, hi=(5, 3), eq="equal letters take the diagonal + 1"),
        line_step("one LCS", ["a c e"], "the corner is the length, not the string"),
    ]
    titles = {"memo": "LCS — memo", "tab": "LCS — table", "row": "LCS — one row"}
    return _pack(titles[kind], "abcde / ace", steps, 3)


def _ascii(kind):
    dp = ascii_table("sea", "eat")
    assert dp[-1][-1] == 231
    if kind == "rec":
        steps = [line_step("delete the unmatched", ["s costs 115", "t costs 116", "ea is kept"], "115 + 116 = 231")]
        return _pack("ASCII delete — recursion", "sea / eat", steps, 231)
    steps = [
        grid_step("delete cost", dp, hi=(3, 3), eq="equal letters are free; otherwise pay the cheaper deletion"),
    ]
    titles = {"memo": "ASCII delete — memo", "tab": "ASCII delete — table", "wt": "ASCII delete — heaviest kept"}
    return _pack(titles[kind], "sea / eat", steps, 231)


def _edit(kind):
    dp = edit_table("horse", "ros")
    assert dp[-1][-1] == 3
    if kind == "rec":
        steps = [line_step("three operations", ["replace h → r", "delete r", "delete e"], "cost 3")]
        return _pack("Edit distance — recursion", "horse / ros", steps, 3)
    steps = [grid_step("edit table", dp, hi=(5, 3), eq="keep is free; insert, delete, replace cost 1")]
    titles = {"memo": "Edit distance — memo", "tab": "Edit distance — table", "row": "Edit distance — one row"}
    return _pack(titles[kind], "horse / ros", steps, 3)


def _scs(kind):
    text = scs_string("abac", "cab")
    assert text == "cabac"
    steps = [
        line_step("walk the LCS table", ["length 4 + 3 − 2 = 5", text], "every letter of either string is written once; shared letters once"),
    ]
    titles = {"rec": "SCS — recursion", "memo": "SCS — memo", "tab": "SCS — length", "walk": "SCS — reconstruction"}
    return _pack(titles[kind], "abac / cab", steps, text)


def _dist(kind):
    dp = dist_table("rabbbit", "rabbit")
    assert dp[-1][-1] == 3
    steps = [
        row_step("ways to spell rabbit", [("dp last row", dp[-1], len(dp[-1]) - 1)], "three choices of which b to drop"),
    ]
    titles = {"rec": "Embeddings — recursion", "memo": "Embeddings — memo", "tab": "Embeddings — table", "row": "Embeddings — one row"}
    return _pack(titles[kind], "rabbbit / rabbit", steps, 3)


def _inter(kind):
    dp = inter_table("aabcc", "dbbca", "aadbbcbcac")
    assert dp[-1][-1] is True
    steps = [grid_step("can the prefix be coloured?", _grid_bits(dp), hi=(5, 5), eq="last letter comes from s1 or from s2")]
    titles = {"rec": "Interleaving — recursion", "memo": "Interleaving — memo", "tab": "Interleaving — table", "row": "Interleaving — one row"}
    return _pack(titles[kind], "aabcc + dbbca", steps, True)


def _wild(kind):
    dp = wild_table("aa", "*")
    assert dp[-1][-1] is True
    steps = [grid_step("star eats the text", _grid_bits(dp), hi=(2, 1), eq="* is empty (left) or one more character (up)")]
    titles = {"rec": "Wildcard — recursion", "memo": "Wildcard — memo", "tab": "Wildcard — table", "star": "Wildcard — star column"}
    return _pack(titles[kind], "aa / *", steps, True)


def _regex(kind):
    dp = regex_table("aa", "a*")
    assert dp[-1][-1] is True
    steps = [grid_step("a* is one token", _grid_bits(dp), hi=(2, 2), eq="zero copies jumps two columns; one more copy stays in the column")]
    titles = {"rec": "Regex — recursion", "memo": "Regex — memo", "tab": "Regex — table", "star": "Regex — x* token"}
    return _pack(titles[kind], "aa / a*", steps, True)


def _brack(kind):
    answer = bracket_count(5, "()))()")
    assert answer == 5
    steps = [line_step("regular and contains s", ["length 10, pattern ()))()", "only the absorbing KMP state at balance 0"], "5 sequences")]
    titles = {"rec": "Brackets — recursion", "memo": "Brackets — memo", "tab": "Brackets — layers", "kmp": "Brackets — KMP state"}
    return _pack(titles[kind], "n = 5, s = ()))()", steps, answer)


def _good(kind):
    answer = good_count(2, "aa", "da", "b")
    assert answer == 51
    steps = [chip_step("between aa and da, no b", [("aa..az except ab", 25, "green"), ("ca..cz except cb", 25, "green"), ("da", 1, "indigo")], "25 + 25 + 1 = 51")]
    titles = {"rec": "Good strings — bounds", "memo": "Good strings — memo", "tab": "Good strings — digits", "kmp": "Good strings — avoid evil"}
    return _pack(titles[kind], "aa .. da, evil b", steps, answer)


def stories():
    return {
        ("lcs", "rec"): _lcs("rec"),
        ("lcs", "memo"): _lcs("memo"),
        ("lcs", "tab"): _lcs("tab"),
        ("lcs", "row"): _lcs("row"),
        ("ascii", "rec"): _ascii("rec"),
        ("ascii", "memo"): _ascii("memo"),
        ("ascii", "tab"): _ascii("tab"),
        ("ascii", "wt"): _ascii("wt"),
        ("edit", "rec"): _edit("rec"),
        ("edit", "memo"): _edit("memo"),
        ("edit", "tab"): _edit("tab"),
        ("edit", "row"): _edit("row"),
        ("scs", "rec"): _scs("rec"),
        ("scs", "memo"): _scs("memo"),
        ("scs", "tab"): _scs("tab"),
        ("scs", "walk"): _scs("walk"),
        ("dist", "rec"): _dist("rec"),
        ("dist", "memo"): _dist("memo"),
        ("dist", "tab"): _dist("tab"),
        ("dist", "row"): _dist("row"),
        ("inter", "rec"): _inter("rec"),
        ("inter", "memo"): _inter("memo"),
        ("inter", "tab"): _inter("tab"),
        ("inter", "row"): _inter("row"),
        ("wild", "rec"): _wild("rec"),
        ("wild", "memo"): _wild("memo"),
        ("wild", "tab"): _wild("tab"),
        ("wild", "star"): _wild("star"),
        ("regex", "rec"): _regex("rec"),
        ("regex", "memo"): _regex("memo"),
        ("regex", "tab"): _regex("tab"),
        ("regex", "star"): _regex("star"),
        ("brack", "rec"): _brack("rec"),
        ("brack", "memo"): _brack("memo"),
        ("brack", "tab"): _brack("tab"),
        ("brack", "kmp"): _brack("kmp"),
        ("good", "rec"): _good("rec"),
        ("good", "memo"): _good("memo"),
        ("good", "tab"): _good("tab"),
        ("good", "kmp"): _good("kmp"),
    }


def get(problem, form):
    return stories()[(problem, form)]


if __name__ == "__main__":
    data = stories()
    expect = {
        "lcs": "3", "ascii": "231", "edit": "3", "scs": "cabac", "dist": "3",
        "inter": "true", "wild": "true", "regex": "true", "brack": "5", "good": "51",
    }
    for key, (title, inp, steps, final) in data.items():
        assert final == expect[key[0]], (key, final)
        assert 1 <= len(steps) <= 8, (key, len(steps))
        print(f"{key[0]:6} {key[1]:5} steps={len(steps)} final={final}")
    assert lcs_len("abc", "abc") == 3 and lcs_len("abc", "def") == 0
    assert ascii_table("delete", "leet")[-1][-1] == 403
    assert edit_table("intention", "execution")[-1][-1] == 5
    assert edit_table("", "abc")[-1][-1] == 3
    assert scs_string("aaaaaaaa", "aaaaaaaa") == "aaaaaaaa"
    assert dist_table("babgbag", "bag")[-1][-1] == 5
    assert inter_table("aabcc", "dbbca", "aadbbbaccc")[-1][-1] is False
    assert inter_table("", "", "")[-1][-1] is True
    assert wild_table("aa", "a")[-1][-1] is False
    assert wild_table("cb", "?a")[-1][-1] is False
    assert wild_table("adceb", "*a*b")[-1][-1] is True
    assert wild_table("acdcb", "a*c?b")[-1][-1] is False
    assert regex_table("aa", "a")[-1][-1] is False
    assert regex_table("ab", ".*")[-1][-1] is True
    assert bracket_count(3, "(()") == 4
    assert bracket_count(2, "(((") == 0
    assert good_count(8, "leetcode", "leetgoes", "leet") == 0
    assert good_count(2, "gx", "gz", "x") == 2
    print("keys", len(data))
