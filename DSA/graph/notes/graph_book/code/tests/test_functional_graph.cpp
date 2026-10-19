// test_functional_graph.cpp -- cycle decomposition against direct simulation of f:
// v is on a cycle iff f^k(v) = v for some 1 <= k <= n; dist_to_cycle = first step on a
// cycle; entry = f^dist(v); cycles listed in arc order; ids consistent.
#define TEST_NAME "functional_graph"
#include "check.hpp"
#include "../lib/functional_graph.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 5000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 12);
        auto f = g.functional(n);
        g_case = "f =";
        for (int x : f) g_case += " " + to_string(x);
        FunctionalGraph fg;
        fg.build(f);
        vector<char> oncyc(n, 0);
        for (int v = 0; v < n; v++) {
            int x = f[v];
            for (int k = 1; k <= n; k++, x = f[x]) if (x == v) { oncyc[v] = 1; break; }
        }
        for (int v = 0; v < n; v++) {
            int x = v, d = 0;
            while (!oncyc[x]) { x = f[x]; d++; }
            CHECK(fg.dist_to_cycle[v] == d, "dist_to_cycle[" << v << "] = " << fg.dist_to_cycle[v] << ", expected " << d);
            CHECK(fg.entry[v] == x, "entry[" << v << "]");
            CHECK(fg.cycle_id[v] == fg.cycle_id[x], "cycle_id differs from its entry");
            CHECK((fg.pos[v] != -1) == (bool)oncyc[v], "pos[] only for cycle vertices");
        }
        int total = 0;
        for (int c = 0; c < (int)fg.cycles.size(); c++) {
            auto& cy = fg.cycles[c];
            int L = cy.size();
            total += L;
            for (int i = 0; i < L; i++) {
                CHECK(f[cy[i]] == cy[(i + 1) % L], "cycle not in arc order");
                CHECK(fg.cycle_id[cy[i]] == c && fg.pos[cy[i]] == i, "cycle_id / pos");
            }
        }
        int expected = 0;
        for (int v = 0; v < n; v++) expected += oncyc[v];
        CHECK(total == expected, "every cycle vertex in exactly one cycle");
        // the k-th iterate formula from the header comment
        for (int v = 0; v < n; v++) {
            int x = v;
            for (int k = 0; k <= 2 * n; k++, x = f[x]) {
                if (k < fg.dist_to_cycle[v]) continue;
                auto& cy = fg.cycles[fg.cycle_id[v]];
                int L = cy.size();
                CHECK(cy[(fg.pos[fg.entry[v]] + k - fg.dist_to_cycle[v]) % L] == x, "f^k formula");
            }
        }
    }
    report_ok(cases);
}
