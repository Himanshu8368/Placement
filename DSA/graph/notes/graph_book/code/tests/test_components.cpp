// test_components.cpp -- component labels against Floyd-Warshall reachability: same id iff
// mutually reachable; ids numbered by smallest vertex; sizes correct.
#define TEST_NAME "components"
#include "check.hpp"
#include "../lib/components.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 5000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 12), m = g.rnd(0, 14);
        auto e = g.graph(n, m, true, true);
        g_case = dump_edges(n, e);
        auto reach = closure(n, e, false);
        Components c;
        c.run(to_adj(n, e, false));
        for (int u = 0; u < n; u++)
            for (int v = 0; v < n; v++)
                CHECK((c.comp[u] == c.comp[v]) == (bool)reach[u][v], "comp[" << u << "] vs comp[" << v << "]");
        int next = 0;
        vector<int> cnt(c.count, 0);
        for (int v = 0; v < n; v++) {
            CHECK(c.comp[v] >= 0 && c.comp[v] < c.count, "id range");
            if (c.comp[v] == next) next++;
            else CHECK(c.comp[v] < next, "ids must appear in order of the smallest vertex");
            cnt[c.comp[v]]++;
        }
        CHECK(next == c.count && cnt == c.size, "count / sizes");
    }
    report_ok(cases);
}
