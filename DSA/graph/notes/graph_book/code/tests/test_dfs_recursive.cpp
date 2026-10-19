// test_dfs_recursive.cpp -- RecDFS: visited set = Floyd-Warshall reachability, parent arcs
// exist, tin/tout satisfy the parenthesis and white-path properties.
#define TEST_NAME "dfs_recursive"
#include "check.hpp"
#include "../lib/dfs_recursive.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 4000; seed++, cases++) {
        Gen g(seed * 7919ULL);
        int n = g.rnd(1, 9), m = g.rnd(0, 16);
        bool directed = g.rnd(0, 1);
        auto e = g.graph(n, m, true, true, directed);
        g_case = dump_edges(n, e) + (directed ? " directed" : " undirected");
        auto adj = to_adj(n, e, directed);
        auto reach = closure(n, e, directed);
        int root = g.rnd(-1, n - 1);
        RecDFS d;
        d.run(adj, root);
        check_dfs(d, adj, reach, root);
        // number of DFS trees = number of roots chosen
        if (root == -1 && !directed) {
            auto lab = naive_labels(n, e);
            int comps = 0, roots = 0;
            for (int v = 0; v < n; v++) { comps += lab[v] == v; roots += d.parent[v] == -1; }
            CHECK(comps == roots, "one DFS tree per component");
        }
    }
    report_ok(cases);
}
