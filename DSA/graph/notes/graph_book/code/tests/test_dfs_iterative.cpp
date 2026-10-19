// test_dfs_iterative.cpp -- IterDFS: visited set = Floyd-Warshall reachability, parent arcs
// exist, tin/tout satisfy the parenthesis and white-path properties, and the output is
// identical to the recursive DFS (same order, parents and times).
#define TEST_NAME "dfs_iterative"
#include "check.hpp"
#include "../lib/dfs_iterative.hpp"
#include "../lib/dfs_recursive.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 4000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 9), m = g.rnd(0, 16);
        bool directed = g.rnd(0, 1);
        auto e = g.graph(n, m, true, true, directed);
        g_case = dump_edges(n, e) + (directed ? " directed" : " undirected");
        auto adj = to_adj(n, e, directed);
        auto reach = closure(n, e, directed);
        int root = g.rnd(-1, n - 1);                    // -1 = whole forest
        IterDFS it;
        it.run(adj, root);
        check_dfs(it, adj, reach, root);
        RecDFS rec;
        rec.run(adj, root);
        CHECK(it.order == rec.order && it.parent == rec.parent && it.tin == rec.tin && it.tout == rec.tout,
              "iterative and recursive DFS differ");
        for (int u = 0; u < n; u++)
            for (int v = 0; v < n; v++) {
                bool anc = false;                       // naive: climb parents from v
                if (it.tin[v] != -1)
                    for (int x = v; x != -1; x = it.parent[x]) if (x == u) anc = true;
                CHECK(it.is_ancestor(u, v) == anc, "is_ancestor(" << u << "," << v << ")");
            }
    }
    // a path of 10^6 vertices: no recursion, so no stack overflow
    int n = 1000000;
    vector<vector<int>> adj(n);
    for (int i = 0; i + 1 < n; i++) { adj[i].push_back(i + 1); adj[i + 1].push_back(i); }
    IterDFS it;
    it.run(adj, 0);
    g_case = "path of 10^6 vertices";
    CHECK(it.tout[0] == 2 * n - 1 && it.parent[n - 1] == n - 2, "long path");
    report_ok(cases + 1);
}
