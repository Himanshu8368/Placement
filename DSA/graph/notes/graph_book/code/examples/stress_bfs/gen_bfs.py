"""Random small graph for the BFS stress test: python gen_bfs.py SEED.
Small sizes first (n grows slowly with the seed), so the first failure is a small one."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gen"))
from gen import Gen  # noqa: E402

seed = int(sys.argv[1])
g = Gen(seed)
n = g.rnd(1, 2 + seed // 50)                       # n <= 2 for seeds 1..49, then larger
m = g.rnd(0, n * (n - 1) // 2)
edges = g.graph(n, m)                              # simple undirected graph
print(n, len(edges))
for u, v in edges:
    print(u + 1, v + 1)
