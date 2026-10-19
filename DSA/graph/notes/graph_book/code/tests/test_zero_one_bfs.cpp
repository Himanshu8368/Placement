// test_zero_one_bfs.cpp -- 0-1 BFS against exhaustive search over simple paths, and the
// par[] pointers form a shortest-path tree.
#define TEST_NAME "zero_one_bfs"
#include "check.hpp"
#include "../lib/zero_one_bfs.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 4000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 7), m = g.rnd(0, 14);
        bool directed = g.rnd(0, 1);
        auto we = g.weighted(g.graph(n, m, true, true, directed), 0, 1);
        int s = g.rnd(0, n - 1);
        g_case = dump_wedges(n, we) + (directed ? " directed" : " undirected") + " s=" + to_string(s);
        vector<vector<pair<int, int>>> adj(n);
        for (auto& x : we) {
            adj[x.u].push_back({x.v, (int)x.w});
            if (!directed) adj[x.v].push_back({x.u, (int)x.w});
        }
        SimplePaths sp(n, we, directed);
        ZeroOneBFS z;
        z.run(adj, s);
        for (int t = 0; t < n; t++) {
            long long expect = sp.best[s][t];
            if (expect >= BR_INF) { CHECK(z.dist[t] == ZeroOneBFS::INF, "reached an unreachable vertex " << t); continue; }
            CHECK(z.dist[t] == expect, "dist[" << t << "] = " << z.dist[t] << ", expected " << expect);
            if (t != s) {
                int p = z.par[t];
                bool ok = false;
                for (auto& e : adj[p]) ok |= e.first == t && z.dist[p] + e.second == z.dist[t];
                CHECK(ok, "par[" << t << "] is not on a shortest path");
            }
        }
    }
    report_ok(cases);
}
