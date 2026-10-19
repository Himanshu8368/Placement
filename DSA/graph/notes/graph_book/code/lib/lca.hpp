#pragma once
// lca.hpp -- lowest common ancestor by binary lifting.
//   build(adj, root): adj is a TREE (undirected, n - 1 edges); BFS from root, so no
//                     recursion; up[j][v] = 2^j-th ancestor of v (root's ancestor = root).
//   kth_ancestor(v, k): the k-th ancestor of v, or -1 if k > depth[v];
//   lca(u, v)         : the lowest common ancestor;
//   dist(u, v)        : number of edges on the u-v path.
// Complexity: O(n log n) build and memory, O(log n) per query.
// Indexing  : vertices 0..n-1.
// Pitfalls  : lift the deeper vertex to the same depth FIRST; then lift both while their
//             ancestors differ and answer up[0][u]; LOG must satisfy 2^LOG > n.
#include <bits/stdc++.h>
using namespace std;

struct LCA {
    int n = 0, LOG = 1;
    vector<vector<int>> up;
    vector<int> depth;

    void build(const vector<vector<int>>& adj, int root = 0) {
        n = adj.size();
        LOG = 1;
        while ((1 << LOG) <= n) LOG++;
        up.assign(LOG, vector<int>(n, root));
        depth.assign(n, -1);
        vector<int> q(1, root);
        depth[root] = 0;
        for (size_t head = 0; head < q.size(); head++) {
            int u = q[head];
            for (int v : adj[u])
                if (depth[v] == -1) {
                    depth[v] = depth[u] + 1;
                    up[0][v] = u;
                    q.push_back(v);
                }
        }
        for (int j = 1; j < LOG; j++)
            for (int v = 0; v < n; v++) up[j][v] = up[j - 1][up[j - 1][v]];
    }

    int kth_ancestor(int v, int k) const {
        if (k > depth[v]) return -1;
        for (int j = 0; j < LOG; j++)
            if (k >> j & 1) v = up[j][v];
        return v;
    }

    int lca(int u, int v) const {
        if (depth[u] < depth[v]) swap(u, v);
        u = kth_ancestor(u, depth[u] - depth[v]);   // same depth now
        if (u == v) return u;
        for (int j = LOG - 1; j >= 0; j--)
            if (up[j][u] != up[j][v]) { u = up[j][u]; v = up[j][v]; }
        return up[0][u];
    }

    int dist(int u, int v) const { return depth[u] + depth[v] - 2 * depth[lca(u, v)]; }
};
