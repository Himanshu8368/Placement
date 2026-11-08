"""
gen.py -- the generators of gen.hpp in Python, for scripting and for stress.py.

As a module:
    from gen import Gen
    g = Gen(seed); a = g.array(8, -5, 5)

As a command line (prints contest input: n, then the values):
    python gen.py array      N LO HI      SEED
    python gen.py sorted     N LO HI      SEED
    python gen.py distinct   N LO HI      SEED
    python gen.py perm       N            SEED
    python gen.py few        N K          SEED      (N values drawn from K distinct values)
    python gen.py intervals  N LO HI      SEED      (N, then N lines "l r")
    python gen.py matrix     R C LO HI    SEED      (R C, then R rows)
  add --k LO HI to append a random integer k in [LO, HI] on its own line after the array.
Small value ranges are the best bug finders: they create duplicates, zeros and ties.
"""
import random
import sys


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def rnd(self, lo, hi):
        return self.r.randint(lo, hi)

    def array(self, n, lo, hi):
        return [self.rnd(lo, hi) for _ in range(n)]

    def sorted(self, n, lo, hi):
        return sorted(self.array(n, lo, hi))

    def distinct(self, n, lo, hi):
        a = self.r.sample(range(lo, hi + 1), n)
        return a

    def perm(self, n, base=1):
        p = list(range(base, base + n))
        self.r.shuffle(p)
        return p

    def few(self, n, k):
        pool = self.array(k, -5, 5)
        return [self.r.choice(pool) for _ in range(n)]

    def intervals(self, n, lo, hi):
        out = []
        for _ in range(n):
            x, y = self.rnd(lo, hi), self.rnd(lo, hi)
            out.append((min(x, y), max(x, y)))
        return out

    def matrix(self, r, c, lo, hi):
        return [self.array(c, lo, hi) for _ in range(r)]


def main():
    args = [x for x in sys.argv[1:]]
    kopt = None
    if '--k' in args:
        i = args.index('--k')
        kopt = (int(args[i + 1]), int(args[i + 2]))
        del args[i:i + 3]
    kind, nums = args[0], [int(x) for x in args[1:]]
    g = Gen(nums[-1])
    out = []
    if kind in ('array', 'sorted', 'distinct'):
        a = getattr(g, kind)(nums[0], nums[1], nums[2])
        out = [str(len(a)), ' '.join(map(str, a))]
    elif kind == 'perm':
        a = g.perm(nums[0])
        out = [str(len(a)), ' '.join(map(str, a))]
    elif kind == 'few':
        a = g.few(nums[0], nums[1])
        out = [str(len(a)), ' '.join(map(str, a))]
    elif kind == 'intervals':
        v = g.intervals(nums[0], nums[1], nums[2])
        out = [str(len(v))] + ['%d %d' % p for p in v]
    elif kind == 'matrix':
        m = g.matrix(nums[0], nums[1], nums[2], nums[3])
        out = ['%d %d' % (nums[0], nums[1])] + [' '.join(map(str, row)) for row in m]
    else:
        sys.exit('unknown kind ' + kind)
    if kopt:
        out.append(str(g.rnd(*kopt)))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
