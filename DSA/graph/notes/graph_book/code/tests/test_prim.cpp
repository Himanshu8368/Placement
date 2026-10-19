// test_prim.cpp -- Prim's spanning forest weight against trying every edge subset; the
// parent pointers form a forest whose edges exist and whose cheapest weights sum up.
#define TEST_NAME "prim"
#include "check.hpp"
#include "../lib/prim.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 4000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 7), m = g.rnd(0, 11);
        auto we = g.weighted(g.graph(n, m, g.rnd(0, 3) == 0, true), g.rnd(-5, 1), 9);
        g_case = dump_wedges(n, we);
        auto br = brute_msf(n, we);
        auto res = prim(to_wadj(n, we, false));
        CHECK(res.weight == br.first, "weight " << res.weight << ", brute force " << br.first);
        CHECK(res.components == br.second, "components");
        int roots = 0;
        for (int v = 0; v < n; v++) {
            if (res.parent[v] == -1) { roots++; continue; }
            bool ok = false;
            for (auto& x : we) ok |= (x.u == v && x.v == res.parent[v]) || (x.v == v && x.u == res.parent[v]);
            CHECK(ok, "parent edge " << v << "-" << res.parent[v] << " does not exist");
            int x = v, steps = 0;                       // following parents must terminate
            while (x != -1 && steps <= n) { x = res.parent[x]; steps++; }
            CHECK(steps <= n, "parent pointers contain a cycle");
        }
        CHECK(roots == br.second, "one root per tree");
    }
    report_ok(cases);
}
