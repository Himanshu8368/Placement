"""Throwaway validation for Chapter 24 Part A engines. Mirrors what the chapter cells will run."""
import random
from collections import Counter

random.seed(0)

# ---------------- A1: classic Mo, count distinct in [l,r] inclusive ----------------
def mo_distinct(a, qs, B, count_move=False):
    n = len(a)
    m = {}
    for v in a:
        m.setdefault(v, len(m))
    b = [m[v] for v in a]
    freq = [0] * len(m)
    distinct = 0
    move = 0
    cl, cr = 0, -1

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

    order = sorted(range(len(qs)), key=lambda i: (qs[i][0] // B, qs[i][1]))
    ans = [0] * len(qs)
    prev = (0, -1)
    for i in order:
        l, r = qs[i]
        if count_move:
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
    if count_move:
        return ans, move
    return ans


def distinct_brute(a, l, r):
    return len(set(a[l:r + 1]))


for _ in range(200):
    n = random.randint(1, 60)
    a = [random.randint(0, 5) for _ in range(n)]
    q = random.randint(1, 25)
    qs = [tuple(sorted((random.randrange(n), random.randrange(n)))) for _ in range(q)]
    B = max(1, int(n / (q ** 0.5))) if q else 1
    got = mo_distinct(a, qs, B)
    exp = [distinct_brute(a, l, r) for l, r in qs]
    assert got == exp, (a, qs, got, exp)
print("A1 distinct stress OK")


# ---------------- A2: rollback Mo, count equal pairs in [l,r] ----------------
def pairs_in(a, l, r):
    c = Counter(a[l:r + 1])
    return sum(x * (x - 1) // 2 for x in c.values())


def rollback_pairs(a, qs, B):
    n = len(a)
    m = {}
    for v in a:
        m.setdefault(v, len(m))
    b = [m[v] for v in a]
    freq = [0] * len(m)
    pairs = 0
    hist = []

    def add(i):
        nonlocal pairs
        v = b[i]; hist.append((v, freq[v])); pairs += freq[v]; freq[v] += 1

    def snapshot():
        return len(hist), pairs

    def rollback(s):
        nonlocal pairs
        hh, pp = s
        while len(hist) > hh:
            v, old = hist.pop(); freq[v] = old
        pairs = pp

    ans = [0] * len(qs)
    order = sorted(range(len(qs)), key=lambda i: (qs[i][0] // B, qs[i][1]))
    last_block = -1
    R = -1
    for i in order:
        l, r = qs[i]
        bl = l // B
        block_end = min(n - 1, (bl + 1) * B - 1)
        if bl != last_block:
            rollback((0, 0))
            R = block_end
            last_block = bl
        while R < r:
            R += 1; add(R)
        if r <= block_end:
            ans[i] = pairs_in(a, l, r)
            continue
        s = snapshot()
        for j in range(block_end, l - 1, -1):
            add(j)
        ans[i] = pairs
        rollback(s)
    return ans


for _ in range(200):
    n = random.randint(1, 60)
    a = [random.randint(0, 4) for _ in range(n)]
    q = random.randint(1, 25)
    qs = [tuple(sorted((random.randrange(n), random.randrange(n)))) for _ in range(q)]
    B = max(1, int(round(n ** 0.5)))
    got = rollback_pairs(a, qs, B)
    exp = [pairs_in(a, l, r) for l, r in qs]
    assert got == exp, (a, qs, got, exp)
print("A2 rollback pairs stress OK")


# ---------------- A3: Mo with updates, count distinct in [l,r] after t updates ----------------
def mo_with_updates(a0, queries, updates):
    a = list(a0)
    n = len(a)
    m = {}

    def id_of(v):
        m.setdefault(v, len(m))
        return m[v]

    b = [id_of(v) for v in a]
    for pos, x in updates:
        id_of(x)
    freq = [0] * (len(m) + 1)
    distinct = 0
    cl, cr, ct = 0, -1, 0

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

    ups = [[pos, x, None] for pos, x in updates]

    def apply_time():
        nonlocal ct, distinct
        pos, x = ups[ct][0], ups[ct][1]
        ups[ct][2] = a[pos]
        nid = id_of(x)
        if cl <= pos <= cr:
            freq[b[pos]] -= 1
            if freq[b[pos]] == 0:
                distinct -= 1
            freq[nid] += 1
            if freq[nid] == 1:
                distinct += 1
        a[pos] = x
        b[pos] = nid
        ct += 1

    def revert_time():
        nonlocal ct, distinct
        ct -= 1
        pos, x, old = ups[ct]
        nid = b[pos]
        oid = id_of(old)
        if cl <= pos <= cr:
            freq[nid] -= 1
            if freq[nid] == 0:
                distinct -= 1
            freq[oid] += 1
            if freq[oid] == 1:
                distinct += 1
        a[pos] = old
        b[pos] = oid

    B = max(1, int(round(n ** (2 / 3))))
    order = sorted(range(len(queries)), key=lambda i: (queries[i][0] // B, queries[i][1] // B, queries[i][2]))
    ans = [0] * len(queries)
    for i in order:
        l, r, t = queries[i]
        while cr < r:
            cr += 1; add(cr)
        while cl > l:
            cl -= 1; add(cl)
        while cl < l:
            remove(cl); cl += 1
        while cr > r:
            remove(cr); cr -= 1
        while ct < t:
            apply_time()
        while ct > t:
            revert_time()
        ans[i] = distinct
    return ans


def distinct_brute_t(a0, updates, l, r, t):
    arr = list(a0)
    for k in range(t):
        pos, x = updates[k]
        arr[pos] = x
    return len(set(arr[l:r + 1]))


for _ in range(200):
    n = random.randint(1, 40)
    a = [random.randint(0, 5) for _ in range(n)]
    u = random.randint(0, 15)
    updates = [(random.randrange(n), random.randint(0, 5)) for _ in range(u)]
    q = random.randint(1, 15)
    queries = [(*sorted((random.randrange(n), random.randrange(n))), random.randint(0, u)) for _ in range(q)]
    got = mo_with_updates(a, queries, updates)
    exp = [distinct_brute_t(a, updates, l, r, t) for l, r, t in queries]
    assert got == exp, (a, updates, queries, got, exp)
print("A3 Mo with updates stress OK")
