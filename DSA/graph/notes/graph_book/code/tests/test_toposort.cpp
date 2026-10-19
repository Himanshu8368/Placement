// test_toposort.cpp -- Kahn and DFS orders: empty iff the graph has a cycle (closure), and
// otherwise a permutation in which every arc points forward.
#define TEST_NAME "toposort"
#include "check.hpp"
#include "../lib/toposort.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 5000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 10), m = g.rnd(0, 16);
        auto e = g.rnd(0, 1) ? g.dag(n, m) : g.digraph(n, m, g.rnd(0, 3) == 0, g.rnd(0, 1));
        g_case = dump_edges(n, e) + " directed";
        auto adj = to_adj(n, e, true);
        auto sc = strict_closure(n, e);
        bool cyclic = false;
        for (int v = 0; v < n; v++) cyclic |= sc[v][v] == 1;
        vector<vector<int>> answers = {topo_kahn(adj), topo_dfs(adj)};
        for (int which = 0; which < 2; which++) {
            auto& ord = answers[which];
            const char* name = which ? "topo_dfs" : "topo_kahn";
            if (cyclic) { CHECK(ord.empty(), name << " must return empty on a cyclic graph"); continue; }
            CHECK((int)ord.size() == n, name << " returned " << ord.size() << " vertices");
            vector<int> pos(n, -1);
            for (int i = 0; i < n; i++) { CHECK(pos[ord[i]] == -1, "repeated vertex"); pos[ord[i]] = i; }
            for (auto& x : e) CHECK(pos[x.first] < pos[x.second], name << ": arc " << x.first << "->" << x.second << " points backwards");
        }
    }
    report_ok(cases);
}
