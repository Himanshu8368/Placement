"""Throwaway: validate the two P2 rollback engines against brutes."""
import random
from collections import Counter, defaultdict

random.seed(0)


# ---- 703D: XOR of values that appear an even (positive) number of times, via rollback Mo ----
def seven_brute(a, l, r):
    c = Counter(a[l:r + 1])
    x = 0
    for v, cnt in c.items():
        if cnt % 2 == 0:                 # cnt >= 2 automatically since present in the range
            x ^= v
    return x


def seven_rollback(a, qs, B=None):
    n = len(a)
    if B is None:
        B = max(1, int(round(n ** 0.5)))
    comp = {}
    b = []
    for v in a:
        comp.setdefault(v, len(comp))
        b.append(comp[v])
    orig = list(comp.keys())             # id -> original value (for the XOR)
    freq = [0] * len(comp)
    x = 0
    hist = []

    def add(i):
        nonlocal x
        v = b[i]
        hist.append((v, freq[v]))
        if freq[v] >= 1:                 # parity flips from included<->excluded only once freq>=1
            x ^= orig[v]
        freq[v] += 1

    def snapshot():
        return len(hist), x

    def rollback(snap):
        nonlocal x
        hh, xx = snap
        while len(hist) > hh:
            v, old = hist.pop()
            freq[v] = old
        x = xx

    order = sorted(range(len(qs)), key=lambda i: (qs[i][0] // B, qs[i][1]))
    ans = [0] * len(qs)
    last_block, R = -1, -1
    for i in order:
        l, r = qs[i]
        bl = l // B
        block_end = min(n - 1, (bl + 1) * B - 1)
        if bl != last_block:
            rollback((0, 0))
            R = block_end
            last_block = bl
        while R < r:
            R += 1
            add(R)
        if r <= block_end:
            ans[i] = seven_brute(a, l, r)
            continue
        snap = snapshot()
        for j in range(block_end, l - 1, -1):
            add(j)
        ans[i] = x
        rollback(snap)
    return ans


assert seven_rollback([3, 7, 8], [(0, 2)]) == [0]
assert seven_rollback([1, 2, 1, 3, 3, 2, 3],
                      [(3, 6), (3, 4), (0, 2), (0, 6), (0, 4)]) == [0, 3, 1, 3, 2]
for _ in range(300):
    n = random.randint(1, 50)
    a = [random.randint(0, 7) for _ in range(n)]
    qs = [tuple(sorted((random.randrange(n), random.randrange(n)))) for _ in range(15)]
    B = max(1, int(round(n ** 0.5)))
    assert seven_rollback(a, qs, B) == [seven_brute(a, l, r) for l, r in qs]
print("703D rollback OK")


# ---- self-authored: number of DISTINCT values per range, via rollback Mo (no deletes) ----
def distinct_brute(a, l, r):
    return len(set(a[l:r + 1]))


def distinct_rollback(a, qs, B=None):
    n = len(a)
    if B is None:
        B = max(1, int(round(n ** 0.5)))
    comp = {}
    b = []
    for v in a:
        comp.setdefault(v, len(comp))
        b.append(comp[v])
    freq = [0] * len(comp)
    distinct = 0
    hist = []

    def add(i):
        nonlocal distinct
        v = b[i]
        hist.append((v, freq[v]))
        if freq[v] == 0:
            distinct += 1
        freq[v] += 1

    def snapshot():
        return len(hist), distinct

    def rollback(snap):
        nonlocal distinct
        hh, dd = snap
        while len(hist) > hh:
            v, old = hist.pop()
            freq[v] = old
        distinct = dd

    order = sorted(range(len(qs)), key=lambda i: (qs[i][0] // B, qs[i][1]))
    ans = [0] * len(qs)
    last_block, R = -1, -1
    for i in order:
        l, r = qs[i]
        bl = l // B
        block_end = min(n - 1, (bl + 1) * B - 1)
        if bl != last_block:
            rollback((0, 0))
            R = block_end
            last_block = bl
        while R < r:
            R += 1
            add(R)
        if r <= block_end:
            ans[i] = distinct_brute(a, l, r)
            continue
        snap = snapshot()
        for j in range(block_end, l - 1, -1):
            add(j)
        ans[i] = distinct
        rollback(snap)
    return ans


assert distinct_rollback([1, 2, 1, 3, 3, 2, 3], [(0, 6), (0, 2), (3, 5)]) == [3, 2, 2]
for _ in range(300):
    n = random.randint(1, 50)
    a = [random.randint(0, 7) for _ in range(n)]
    qs = [tuple(sorted((random.randrange(n), random.randrange(n)))) for _ in range(15)]
    B = max(1, int(round(n ** 0.5)))
    assert distinct_rollback(a, qs, B) == [distinct_brute(a, l, r) for l, r in qs]
print("distinct rollback OK")
