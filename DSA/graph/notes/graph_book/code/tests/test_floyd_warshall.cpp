// test_floyd_warshall.cpp -- all-pairs distances against exhaustive simple paths (when no
// negative cycle exists) and negative-cycle detection against simple-cycle enumeration.
#define TEST_NAME "floyd_warshall"
#include "check.hpp"
#include "../lib/floyd_warshall.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 5000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 6), m = g.rnd(0, 12);
        bool directed = g.rnd(0, 2) != 0;
        int lo = directed ? (int)g.rnd(-5, 0) : 0;      // undirected: non-negative only
        auto we = g.weighted(directed ? g.digraph(n, m, g.rnd(0, 3) == 0, true) : g.graph(n, m, false, true), lo, 9);
        g_case = dump_wedges(n, we) + (directed ? " directed" : " undirected");
        auto d = make_matrix(n);
        for (auto& x : we) {
            d[x.u][x.v] = min(d[x.u][x.v], x.w);
            if (!directed) d[x.v][x.u] = min(d[x.v][x.u], x.w);
        }
        floyd_warshall(d);
        SimplePaths sp(n, we, directed);
        CHECK(has_negative_cycle(d) == sp.any_neg, "has_negative_cycle");
        if (sp.any_neg) continue;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                long long expect = sp.best[i][j] >= BR_INF ? FW_INF : sp.best[i][j];
                CHECK(d[i][j] == expect, "d[" << i << "][" << j << "] = " << d[i][j] << ", expected " << expect);
            }
    }
    report_ok(cases);
}
