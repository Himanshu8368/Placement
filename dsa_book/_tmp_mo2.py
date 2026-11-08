"""Throwaway validation: A4 Hilbert order + A5 freq-of-freq. Uses random.seed(0) shared-kernel style."""
import random
from collections import Counter

random.seed(0)


def hilbert_order(x, y, pow=21, rot=0):
    if pow == 0:
        return 0
    hpow = 1 << (pow - 1)
    if x < hpow:
        seg = 0 if y < hpow else 3
    else:
        seg = 1 if y < hpow else 2
    seg = (seg + rot) % 4
    rotate_delta = [3, 0, 0, 1]
    nx = x % hpow
    ny = y % hpow
    nrot = (rot + rotate_delta[seg]) % 4
    sub = 1 << (2 * pow - 2)
    ans = seg * sub
    add = hilbert_order(nx, ny, pow - 1, nrot)
    ans += add if seg == 1 or seg == 3 else (sub - add - 1)
    return ans


def mo_move(qs, keyfn):
    order = sorted(range(len(qs)), key=lambda i: keyfn(i, qs))
    move = 0
    prev = (0, -1)
    for i in order:
        l, r = qs[i]
        move += abs(l - prev[0]) + abs(r - prev[1])
        prev = (l, r)
    return move


def block_key(B):
    return lambda i, qs: (qs[i][0] // B, qs[i][1])


def hilb_key(i, qs):
    return hilbert_order(qs[i][0], qs[i][1])


def mo_distinct_hilbert(a, qs):
    """Classic Mo but ordered by Hilbert; returns answers + total movement."""
    n = len(a)
    m = {}
    for v in a:
        m.setdefault(v, len(m))
    b = [m[v] for v in a]
    freq = [0] * len(m)
    distinct = 0
    move = 0
    cl, cr = 0, -1
    prev = (0, -1)

    def add(i):
        nonlocal distinct
        freq[b[i]] += 1
        if freq[b[i]] == 1:
            distinct += 1

    def remove(i):
        nonlocal distinct
        freq[b[i]] -= 1
        if freq[b[i]] == 0:
            distinct -= 1

    order = sorted(range(len(qs)), key=lambda i: hilbert_order(qs[i][0], qs[i][1]))
    ans = [0] * len(qs)
    for i in order:
        l, r = qs[i]
        move += abs(l - prev[0]) + abs(r - prev[1])
        prev = (l, r)
        while cr < r:
            cr += 1; add(cr)
        while cl > l:
            cl -= 1; add(cl)
        while cl < l:
            remove(cl); cl += 1
        while cr > r:
            remove(cr); cr -= 1
        ans[i] = distinct
    return ans, move


# correctness of distinct under Hilbert ordering
def distinct_brute(a, l, r):
    return len(set(a[l:r + 1]))


for _ in range(200):
    n = random.randint(1, 60)
    a = [random.randint(0, 5) for _ in range(n)]
    q = random.randint(1, 25)
    qs = [tuple(sorted((random.randrange(n), random.randrange(n)))) for _ in range(q)]
    got, _ = mo_distinct_hilbert(a, qs)
    exp = [distinct_brute(a, l, r) for l, r in qs]
    assert got == exp
print("A4 Hilbert-order Mo distinct correctness OK")

# measured movement: Hilbert vs classic block-sort, summed over many trials
tot_h = tot_b = 0
rmov = random.Random(12345)
for trial in range(60):
    n = rmov.choice([800, 2000, 4000])
    q = n
    qs = [tuple(sorted((rmov.randrange(n), rmov.randrange(n)))) for _ in range(q)]
    B = max(1, int(n / (q ** 0.5)))
    tot_b += mo_move(qs, block_key(B))
    tot_h += mo_move(qs, hilb_key)
print("block-sort total movement:", tot_b, " hilbert total:", tot_h, " ratio:", tot_h / tot_b)
assert tot_h < tot_b

# A5: freq-of-freq, count values appearing exactly k times
def mo_exact_k(a, qs, B, k):
    n = len(a)
    m = {}
    for v in a:
        m.setdefault(v, len(m))
    b = [m[v] for v in a]
    freq = [0] * len(m)
    ftf = [0] * (n + 5)   # ftf[f] = how many values currently have frequency f
    ans = [0] * len(qs)

    def add(i):
        v = b[i]
        old = freq[v]
        if old > 0:
            ftf[old] -= 1
        freq[v] = old + 1
        ftf[old + 1] += 1

    def remove(i):
        v = b[i]
        old = freq[v]
        ftf[old] -= 1
        freq[v] = old - 1
        if old - 1 > 0:
            ftf[old - 1] += 1

    order = sorted(range(len(qs)), key=lambda i: (qs[i][0] // B, qs[i][1]))
    cl, cr = 0, -1
    for i in order:
        l, r = qs[i]
        while cr < r:
            cr += 1; add(cr)
        while cl > l:
            cl -= 1; add(cl)
        while cl < l:
            remove(cl); cl += 1
        while cr > r:
            remove(cr); cr -= 1
        ans[i] = ftf[k]
    return ans


def exact_k_brute(a, l, r, k):
    c = Counter(a[l:r + 1])
    return sum(1 for x in c.values() if x == k)


for _ in range(200):
    n = random.randint(1, 60)
    a = [random.randint(0, 4) for _ in range(n)]
    q = random.randint(1, 25)
    qs = [tuple(sorted((random.randrange(n), random.randrange(n)))) for _ in range(q)]
    B = max(1, int(n / (q ** 0.5)))
    for k in (1, 2, 3):
        got = mo_exact_k(a, qs, B, k)
        exp = [exact_k_brute(a, l, r, k) for l, r in qs]
        assert got == exp, (k, a, qs, got, exp)
print("A5 freq-of-freq exact-k stress OK")
