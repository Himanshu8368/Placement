"""
gen.py -- the generators of gen.hpp in Python, for scripting and for stress.py.

As a module:
    from gen import Gen
    g = Gen(seed); edges = g.tree(5); w = g.weighted(edges, 1, 9)

As a command line (prints contest input; vertices 1-indexed unless --base 0):
    python gen.py tree       N            SEED
    python gen.py connected  N M          SEED
    python gen.py graph      N M          SEED [--loops] [--multi]
    python gen.py digraph    N M          SEED [--loops] [--multi]
    python gen.py dag        N M          SEED
    python gen.py functional N            SEED          (prints N, then f(1..N))
    python gen.py grid       R C PCT      SEED          (prints R C, then the rows)
  add --weights LO HI to append a weight to every edge.
All vertex labels are shuffled and undirected edges get a random orientation.
"""
import random
import sys


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def rnd(self, lo, hi):
        return self.r.randint(lo, hi)

    def perm(self, n):
        p = list(range(n))
        self.r.shuffle(p)
        return p

    def _shuffle(self, e, flip):
        self.r.shuffle(e)
        if flip:
            e = [(v, u) if self.r.randint(0, 1) else (u, v) for u, v in e]
        return e

    def tree(self, n):
        p = self.perm(n)
        e = [(p[i], p[self.r.randint(0, i - 1)]) for i in range(1, n)]
        return self._shuffle(e, True)

    def connected_graph(self, n, m):
        m = max(n - 1, min(m, n * (n - 1) // 2))
        e = self.tree(n)
        have = {(min(u, v), max(u, v)) for u, v in e}
        while len(e) < m:
            u, v = self.r.randrange(n), self.r.randrange(n)
            if u == v or (min(u, v), max(u, v)) in have:
                continue
            have.add((min(u, v), max(u, v)))
            e.append((u, v))
        return self._shuffle(e, True)

    def graph(self, n, m, loops=False, multi=False, directed=False):
        cap = (n * (n - 1) if directed else n * (n - 1) // 2) + (n if loops else 0)
        if not multi:
            m = min(m, cap)
        if not loops and n < 2:
            m = 0
        e, have = [], set()
        while len(e) < m:
            u, v = self.r.randrange(n), self.r.randrange(n)
            if u == v and not loops:
                continue
            key = (u, v) if directed else (min(u, v), max(u, v))
            if not multi and key in have:
                continue
            have.add(key)
            e.append((u, v))
        return e

    def digraph(self, n, m, loops=False, multi=False):
        return self.graph(n, m, loops, multi, True)

    def dag(self, n, m):
        m = min(m, n * (n - 1) // 2)
        p = self.perm(n)
        e, have = [], set()
        while len(e) < m:
            a, b = self.r.randrange(n), self.r.randrange(n)
            if a == b:
                continue
            a, b = min(a, b), max(a, b)
            if (a, b) in have:
                continue
            have.add((a, b))
            e.append((p[a], p[b]))
        return e

    def functional(self, n):
        return [self.r.randrange(n) for _ in range(n)]

    def grid(self, R, C, wall_pct):
        return ["".join("#" if self.r.randint(1, 100) <= wall_pct else "." for _ in range(C))
                for _ in range(R)]

    def weighted(self, e, lo, hi):
        return [(u, v, self.r.randint(lo, hi)) for u, v in e]


def main(argv):
    flags, args, base, weights = [], [], 1, None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--base":
            base = int(argv[i + 1]); i += 2
        elif a == "--weights":
            weights = (int(argv[i + 1]), int(argv[i + 2])); i += 3
        elif a.startswith("--"):
            flags.append(a); i += 1
        else:
            args.append(a); i += 1
    kind, nums = args[0], [int(x) for x in args[1:]]
    g = Gen(nums[-1])
    out = []
    if kind == "functional":
        n = nums[0]
        out.append(str(n))
        out.append(" ".join(str(x + base) for x in g.functional(n)))
    elif kind == "grid":
        R, C, pct = nums[0], nums[1], nums[2]
        out.append("%d %d" % (R, C))
        out.extend(g.grid(R, C, pct))
    else:
        n = nums[0]
        if kind == "tree":
            e = g.tree(n)
        elif kind == "connected":
            e = g.connected_graph(n, nums[1])
        elif kind in ("graph", "digraph"):
            e = g.graph(n, nums[1], "--loops" in flags, "--multi" in flags, kind == "digraph")
        elif kind == "dag":
            e = g.dag(n, nums[1])
        else:
            sys.exit("unknown kind " + kind)
        out.append("%d %d" % (n, len(e)))
        if weights:
            for u, v, w in g.weighted(e, *weights):
                out.append("%d %d %d" % (u + base, v + base, w))
        else:
            for u, v in e:
                out.append("%d %d" % (u + base, v + base))
    print("\n".join(out))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
