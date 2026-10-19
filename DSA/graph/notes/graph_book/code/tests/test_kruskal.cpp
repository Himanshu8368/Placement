// test_kruskal.cpp -- minimum spanning forest weight against trying every edge subset;
// the chosen edges form a forest with n - components edges and the reported weight.
#define TEST_NAME "kruskal"
#include "check.hpp"
#include "../lib/kruskal.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 4000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 7), m = g.rnd(0, 11);
        auto we = g.weighted(g.graph(n, m, g.rnd(0, 3) == 0, true), g.rnd(-5, 1), 9);
        g_case = dump_wedges(n, we);
        auto br = brute_msf(n, we);
        auto res = kruskal(n, we);
        CHECK(res.weight == br.first, "weight " << res.weight << ", brute force " << br.first);
        CHECK(res.components == br.second, "components");
        CHECK((int)res.chosen.size() == n - br.second, "number of chosen edges");
        vector<int> lab(n);
        iota(lab.begin(), lab.end(), 0);
        long long w = 0;
        for (int i : res.chosen) {
            int a = lab[we[i].u], b = lab[we[i].v];
            CHECK(a != b, "chosen edges contain a cycle");
            for (auto& x : lab) if (x == b) x = a;
            w += we[i].w;
        }
        CHECK(w == res.weight, "sum of chosen edges");
    }
    report_ok(cases);
}
