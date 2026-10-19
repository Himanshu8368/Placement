#pragma once
// toposort.hpp -- topological order of a directed graph, two ways.
//   topo_kahn(adj): repeatedly remove a vertex of in-degree 0 (a queue of "ready" vertices).
//   topo_dfs(adj) : reverse post-order of an iterative DFS.
// Both return an order in which every arc u -> v has u before v, or an EMPTY vector if
// the graph has a cycle (with n >= 1 an empty answer is never a valid order).
// Kahn with a min-heap instead of the queue gives the lexicographically smallest order.
// Complexity: O(n + m).
// Indexing  : vertices 0..n-1.
// Pitfalls  : Kahn: a cycle shows up as "fewer than n vertices output" -- check it;
//             DFS: post-order must be REVERSED; an arc to a vertex still on the stack
//             means a cycle; a self-loop u -> u is a cycle.
#include <bits/stdc++.h>
using namespace std;

inline vector<int> topo_kahn(const vector<vector<int>>& adj) {
    int n = adj.size();
    vector<int> indeg(n, 0), order;
    for (int u = 0; u < n; u++)
        for (int v : adj[u]) indeg[v]++;
    for (int u = 0; u < n; u++)
        if (indeg[u] == 0) order.push_back(u);     // order doubles as the queue
    for (size_t head = 0; head < order.size(); head++)
        for (int v : adj[order[head]])
            if (--indeg[v] == 0) order.push_back(v);
    if ((int)order.size() < n) return {};          // the rest lies on / behind a cycle
    return order;
}

inline vector<int> topo_dfs(const vector<vector<int>>& adj) {
    int n = adj.size();
    vector<int> state(n, 0), post, st;             // 0 new, 1 on stack, 2 done
    vector<size_t> it(n, 0);
    for (int s = 0; s < n; s++) {
        if (state[s]) continue;
        state[s] = 1;
        st.assign(1, s);
        while (!st.empty()) {
            int u = st.back();
            if (it[u] == adj[u].size()) { state[u] = 2; post.push_back(u); st.pop_back(); continue; }
            int v = adj[u][it[u]++];
            if (state[v] == 1) return {};          // arc into the stack: cycle
            if (state[v] == 0) { state[v] = 1; st.push_back(v); }
        }
    }
    reverse(post.begin(), post.end());
    return post;
}
