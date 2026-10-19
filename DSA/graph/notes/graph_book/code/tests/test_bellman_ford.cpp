// test_bellman_ford.cpp -- against exhaustive enumeration of simple paths and simple cycles:
//   dist[v] = -INF  iff  s reaches a vertex of a negative simple cycle that reaches v;
//   otherwise dist[v] = shortest simple path (INF if unreachable).
// find_negative_cycle: non-empty iff any negative cycle exists; the arcs chain up and sum < 0.
#define TEST_NAME "bellman_ford"
#include "check.hpp"
#include "../lib/bellman_ford.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 5000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 6), m = g.rnd(0, 11);
        auto we = g.weighted(g.digraph(n, m, g.rnd(0, 3) == 0, true), g.rnd(-6, 0), 9);
        int s = g.rnd(0, n - 1);
        g_case = dump_wedges(n, we) + " directed s=" + to_string(s);
        vector<pair<int, int>> pe;
        for (auto& x : we) pe.push_back({x.u, x.v});
        auto reach = closure(n, pe, true);
        SimplePaths sp(n, we, true);
        BellmanFord bf;
        bool neg = bf.run(n, we, s);
        bool expect_neg = false;
        for (int t = 0; t < n; t++) {
            bool minus_inf = false;
            for (int c = 0; c < n; c++)
                if (sp.on_neg_cycle[c] && reach[s][c] && reach[c][t]) minus_inf = true;
            expect_neg |= minus_inf;
            long long expect = minus_inf ? -BellmanFord::INF : (reach[s][t] ? sp.best[s][t] : BellmanFord::INF);
            CHECK(bf.dist[t] == expect, "dist[" << t << "] = " << bf.dist[t] << ", expected " << expect);
        }
        CHECK(neg == expect_neg, "run() return value");
        auto cyc = find_negative_cycle(n, we);
        CHECK(!cyc.empty() == sp.any_neg, "find_negative_cycle found=" << !cyc.empty() << " expected " << sp.any_neg);
        long long sum = 0;
        for (size_t i = 0; i < cyc.size(); i++) {
            sum += we[cyc[i]].w;
            CHECK(we[cyc[i]].v == we[cyc[(i + 1) % cyc.size()]].u, "cycle arcs do not chain");
        }
        if (!cyc.empty()) CHECK(sum < 0, "cycle weight " << sum << " is not negative");
    }
    report_ok(cases);
}
