// test_bfs.cpp -- BFS distances against Floyd-Warshall; every reconstructed path is a real
// path of length dist; the pop order is sorted by distance.
#define TEST_NAME "bfs"
#include "check.hpp"
#include "../lib/bfs.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 5000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 10), m = g.rnd(0, 18);
        bool directed = g.rnd(0, 1);
        auto e = g.graph(n, m, true, true, directed);
        int s = g.rnd(0, n - 1);
        g_case = dump_edges(n, e) + (directed ? " directed" : " undirected") + " s=" + to_string(s);
        auto adj = to_adj(n, e, directed);
        auto D = apsp_unweighted(n, e, directed);
        BFS b;
        b.run(adj, s);
        for (int t = 0; t < n; t++) {
            CHECK(b.dist[t] == D[s][t], "dist[" << t << "] = " << b.dist[t] << ", expected " << D[s][t]);
            auto p = b.path(t);
            if (D[s][t] == -1) { CHECK(p.empty(), "path to unreachable vertex"); continue; }
            CHECK((int)p.size() == D[s][t] + 1 && p.front() == s && p.back() == t, "path endpoints/length");
            for (size_t i = 0; i + 1 < p.size(); i++) CHECK(has_arc(adj, p[i], p[i + 1]), "path uses a non-edge");
        }
        for (size_t i = 0; i + 1 < b.order.size(); i++) CHECK(b.dist[b.order[i]] <= b.dist[b.order[i + 1]], "pop order");
    }
    report_ok(cases);
}
