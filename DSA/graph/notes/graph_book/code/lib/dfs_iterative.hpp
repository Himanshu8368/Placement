#pragma once
// dfs_iterative.hpp -- depth-first search with an explicit stack (no recursion).
//   run(adj)        : DFS from every unvisited vertex in increasing order (a DFS forest);
//   run(adj, root)  : DFS from root only; unreached vertices keep tin = -1.
//   order[]         : vertices in entry (pre-)order;  parent[v] = -1 for every root;
//   tin[v], tout[v] : entry / exit times from ONE clock that ticks on both events,
//                     so times are 0..2k-1 and u is an ancestor of v (or u == v) iff
//                     tin[u] <= tin[v] && tout[v] <= tout[u].
// Complexity: O(n + m) time, O(n) extra memory. Works for directed and undirected adj.
// Indexing  : vertices 0..n-1.
// Pitfalls  : the visit order equals the recursive DFS only because it[u] remembers the
//             next neighbour to try; pushing ALL neighbours at once gives a different
//             (non-DFS) order and wrong tin/tout.
#include <bits/stdc++.h>
using namespace std;

struct IterDFS {
    vector<int> order, parent, tin, tout;

    void run(const vector<vector<int>>& adj, int root = -1) {
        int n = adj.size();
        order.clear();
        parent.assign(n, -1);
        tin.assign(n, -1);
        tout.assign(n, -1);
        vector<size_t> it(n, 0);     // it[u] = index of the next neighbour of u to try
        vector<int> st;
        int timer = 0;
        auto launch = [&](int s) {
            tin[s] = timer++;
            order.push_back(s);
            st.push_back(s);
            while (!st.empty()) {
                int u = st.back();
                if (it[u] < adj[u].size()) {
                    int v = adj[u][it[u]++];
                    if (tin[v] == -1) {              // tree edge u -> v: "call" dfs(v)
                        parent[v] = u;
                        tin[v] = timer++;
                        order.push_back(v);
                        st.push_back(v);
                    }
                } else {                             // all neighbours done: "return"
                    tout[u] = timer++;
                    st.pop_back();
                }
            }
        };
        if (root >= 0) launch(root);
        else
            for (int s = 0; s < n; s++)
                if (tin[s] == -1) launch(s);
    }

    bool is_ancestor(int u, int v) const {   // u is an ancestor of v (or u == v)
        return tin[u] != -1 && tin[v] != -1 && tin[u] <= tin[v] && tout[v] <= tout[u];
    }
};
