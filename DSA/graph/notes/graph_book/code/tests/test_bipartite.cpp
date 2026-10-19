// test_bipartite.cpp -- the answer agrees with trying all 2^n colourings; a "yes" comes with
// a proper colouring, a "no" with a genuine odd closed walk that is a simple cycle.
#define TEST_NAME "bipartite"
#include "check.hpp"
#include "../lib/bipartite.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 5000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 10), m = g.rnd(0, 12);
        bool loops = g.rnd(0, 3) == 0;
        auto e = g.graph(n, m, loops, true);
        g_case = dump_edges(n, e);
        auto adj = to_adj(n, e, false);
        bool brute = false;
        for (int mask = 0; mask < (1 << n) && !brute; mask++) {
            bool ok = true;
            for (auto& x : e) if ((mask >> x.first & 1) == (mask >> x.second & 1)) { ok = false; break; }
            brute = ok;
        }
        Bipartite b;
        bool res = b.run(adj);
        CHECK(res == brute, "answer " << res << ", brute force says " << brute);
        if (res) {
            for (auto& x : e) CHECK(b.color[x.first] != b.color[x.second], "edge with equal colours");
        } else {
            auto& c = b.odd_cycle;
            int k = c.size();
            CHECK(k % 2 == 1, "odd_cycle has even length " << k);
            set<int> distinct(c.begin(), c.end());
            CHECK((int)distinct.size() == k, "odd_cycle repeats a vertex");
            for (int i = 0; i < k; i++) CHECK(has_arc(adj, c[i], c[(i + 1) % k]), "odd_cycle uses a non-edge");
        }
    }
    report_ok(cases);
}
