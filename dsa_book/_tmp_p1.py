import random
from collections import defaultdict

random.seed(0)


# ---------------- 86D Powerful array ----------------
def powerful_solve(a, qs, B=None):
    """a 0-indexed; qs (l, r) inclusive 0-indexed; power = sum over value v of freq[v]^2 * v."""
    n = len(a)
    if B is None:
        B = max(1, int(n / max(1, len(qs)) ** 0.5))
    freq = defaultdict(int)
    total = 0
    cl, cr = 0, -1

    def add(i):
        nonlocal total
        v = a[i]
        total += (2 * freq[v] + 1) * v
        freq[v] += 1

    def remove(i):
        nonlocal total
        v = a[i]
        freq[v] -= 1
        total -= (2 * freq[v] + 1) * v

    order = sorted(range(len(qs)), key=lambda i: (qs[i][0] // B, qs[i][1]))
    ans = [0] * len(qs)
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
        ans[i] = total
    return ans


def powerful_brute(a, l, r):
    c = defaultdict(int)
    for i in range(l, r + 1):
        c[a[i]] += 1
    return sum(cnt * cnt * v for v, cnt in c.items())


assert powerful_solve([1, 2, 1], [(0, 1), (0, 2)]) == [3, 6]
assert powerful_solve([1, 1, 2, 2, 1, 3, 1, 1], [(1, 6), (0, 5), (1, 6)]) == [20, 20, 20]
for _ in range(300):
    n = random.randint(1, 40)
    a = [random.randint(1, 8) for _ in range(n)]
    qs = [tuple(sorted((random.randrange(n), random.randrange(n)))) for _ in range(15)]
    assert powerful_solve(a, qs) == [powerful_brute(a, l, r) for l, r in qs]
print("86D powerful OK")


# ---------------- 617E XOR and Favorite Number ----------------
def xorfav_solve(a, k, qs, B=None):
    """a 0-indexed (original array), qs (l, r) inclusive 0-indexed over a.
    Count subarrays [i..j] inside [l,r] with xor == k, via Mo over prefix indices [l, r+1]."""
    n = len(a)
    if B is None:
        B = max(1, int(n / max(1, len(qs)) ** 0.5))
    # prefix xor array p of length n+1; subarray [i..j] (0-idx) = p[j+1]^p[i]
    p = [0] * (n + 1)
    for i in range(n):
        p[i + 1] = p[i] ^ a[i]
    size = 1
    while size <= max(max(p), k):
        size <<= 1
    size <<= 1
    freq = [0] * size
    pairs = 0
    cl, cr = 0, -1   # window over prefix indices, inclusive [cl, cr]

    def add(i):
        nonlocal pairs
        pairs += freq[p[i] ^ k]
        freq[p[i]] += 1

    def remove(i):
        nonlocal pairs
        freq[p[i]] -= 1
        pairs -= freq[p[i] ^ k]

    qwin = [(l, r + 1) for (l, r) in qs]    # prefix index window [l, r+1]
    order = sorted(range(len(qwin)), key=lambda i: (qwin[i][0] // B, qwin[i][1]))
    ans = [0] * len(qwin)
    for i in order:
        l, r = qwin[i]
        while cr < r:
            cr += 1; add(cr)
        while cl > l:
            cl -= 1; add(cl)
        while cl < l:
            remove(cl); cl += 1
        while cr > r:
            remove(cr); cr -= 1
        ans[i] = pairs
    return ans


def xorfav_brute(a, k, l, r):
    cnt = 0
    for i in range(l, r + 1):
        x = 0
        for j in range(i, r + 1):
            x ^= a[j]
            if x == k:
                cnt += 1
    return cnt


assert xorfav_solve([1, 2, 1, 1, 0, 3], 3, [(0, 5), (2, 4)]) == [7, 0]
assert xorfav_solve([1, 1, 1, 1, 1], 1, [(0, 4), (1, 3), (0, 2)]) == [9, 4, 4]
for _ in range(300):
    n = random.randint(1, 30)
    a = [random.randint(0, 7) for _ in range(n)]
    k = random.randint(0, 7)
    qs = [tuple(sorted((random.randrange(n), random.randrange(n)))) for _ in range(12)]
    assert xorfav_solve(a, k, qs) == [xorfav_brute(a, k, l, r) for l, r in qs]
print("617E xorfav OK")
