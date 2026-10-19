// test_dijkstra.cpp -- Dijkstra against a naive Bellman-Ford (n rounds over all arcs) and
// against exhaustive simple paths; reconstructed paths have exactly the reported length.
#define TEST_NAME "dijkstra"
#include "check.hpp"
#include "../lib/dijkstra.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 5000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 7), m = g.rnd(0, 15);
        bool directed = g.rnd(0, 1);
        auto we = g.weighted(g.graph(n, m, true, true, directed), 0, g.rnd(0, 1) ? 3 : 1000000000);
        int s = g.rnd(0, n - 1);
        g_case = dump_wedges(n, we) + (directed ? " directed" : " undirected") + " s=" + to_string(s);
        auto adj = to_wadj(n, we, directed);
        vector<long long> bf(n, BR_INF);                // naive Bellman-Ford
        bf[s] = 0;
        for (int r = 0; r < n; r++)
            for (int u = 0; u < n; u++)
                if (bf[u] < BR_INF)
                    for (auto& e : adj[u]) bf[e.first] = min(bf[e.first], bf[u] + e.second);
        SimplePaths sp(n, we, directed);
        Dijkstra dj;
        dj.run(adj, s);
        for (int t = 0; t < n; t++) {
            CHECK(dj.dist[t] == bf[t] && bf[t] == sp.best[s][t], "dist[" << t << "] = " << dj.dist[t] << ", Bellman-Ford " << bf[t]);
            auto p = dj.path(t);
            if (bf[t] >= BR_INF) { CHECK(p.empty(), "path to unreachable"); continue; }
            CHECK(p.front() == s && p.back() == t, "path endpoints");
            long long len = 0;
            for (size_t i = 0; i + 1 < p.size(); i++) {
                long long w = BR_INF;
                for (auto& e : adj[p[i]]) if (e.first == p[i + 1]) w = min(w, e.second);
                CHECK(w < BR_INF, "path uses a non-edge");
                len += w;
            }
            CHECK(len == dj.dist[t], "path length " << len << " != dist " << dj.dist[t]);
        }
    }
    report_ok(cases);
}
