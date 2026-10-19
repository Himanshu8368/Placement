// test_lca.cpp -- binary lifting against naive climbing (lift the deeper vertex one step at
// a time, then both together) on random trees with a random root.
#define TEST_NAME "lca"
#include "check.hpp"
#include "../lib/lca.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 3000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, seed % 10 == 0 ? 200 : 16);
        auto e = g.tree(n);
        int root = g.rnd(0, n - 1);
        g_case = dump_edges(n, e) + " root=" + to_string(root);
        auto adj = to_adj(n, e, false);
        vector<int> par(n, -1), dep(n, -1), st(1, root);   // naive parents by DFS
        dep[root] = 0;
        while (!st.empty()) {
            int u = st.back(); st.pop_back();
            for (int v : adj[u]) if (dep[v] == -1) { dep[v] = dep[u] + 1; par[v] = u; st.push_back(v); }
        }
        LCA L;
        L.build(adj, root);
        for (int q = 0; q < 60; q++) {
            int u = g.rnd(0, n - 1), v = g.rnd(0, n - 1), k = g.rnd(0, n);
            int a = u, b = v;
            while (dep[a] > dep[b]) a = par[a];
            while (dep[b] > dep[a]) b = par[b];
            while (a != b) { a = par[a]; b = par[b]; }
            CHECK(L.lca(u, v) == a, "lca(" << u << "," << v << ") = " << L.lca(u, v) << ", expected " << a);
            CHECK(L.dist(u, v) == dep[u] + dep[v] - 2 * dep[a], "dist");
            int x = u;
            for (int i = 0; i < k && x != -1; i++) x = par[x];
            CHECK(L.kth_ancestor(u, k) == x, "kth_ancestor(" << u << "," << k << ")");
        }
    }
    report_ok(cases);
}
