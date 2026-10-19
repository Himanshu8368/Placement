#pragma once
// cycle.hpp -- find one cycle (or prove there is none), undirected and directed.
//   find_cycle_undirected(n, edges): edges[i] = {u, v}; multi-edges and self-loops allowed.
//       Returns {verts, ids}: verts[0..k-1] and edge ids ids[0..k-1] with edge ids[i]
//       joining verts[i] and verts[(i+1) % k]; both empty if the graph is a forest.
//       Two parallel edges form a cycle of length 2, a self-loop one of length 1.
//   find_cycle_directed(adj): vertices v0 -> v1 -> ... -> v(k-1) -> v0, empty if acyclic
//       (a self-loop u -> u gives {u}).
// Complexity: O(n + m), iterative DFS (no recursion).
// Indexing  : vertices 0..n-1, edge ids 0..m-1.
// Pitfalls  : undirected: skip the EDGE you came in by, not the parent VERTEX -- with
//             "skip parent vertex" two parallel edges u=v are missed;
//             directed: "visited" is not enough, a cycle needs an edge to a vertex that
//             is still ON THE STACK (colour 1); an edge to a finished vertex is harmless.
#include <bits/stdc++.h>
using namespace std;

struct UndirectedCycle {
    vector<int> verts, ids;
};

inline UndirectedCycle find_cycle_undirected(int n, const vector<pair<int, int>>& edges) {
    vector<vector<pair<int, int>>> adj(n);          // (neighbour, edge id)
    for (int i = 0; i < (int)edges.size(); i++) {
        adj[edges[i].first].push_back({edges[i].second, i});
        if (edges[i].first != edges[i].second) adj[edges[i].second].push_back({edges[i].first, i});
    }
    vector<int> state(n, 0), par(n, -1), pedge(n, -1);   // 0 new, 1 on stack, 2 done
    vector<size_t> it(n, 0);
    vector<int> st;
    for (int s = 0; s < n; s++) {
        if (state[s]) continue;
        state[s] = 1;
        st.assign(1, s);
        while (!st.empty()) {
            int u = st.back();
            if (it[u] == adj[u].size()) { state[u] = 2; st.pop_back(); continue; }
            int v = adj[u][it[u]].first, id = adj[u][it[u]].second;
            it[u]++;
            if (id == pedge[u]) continue;            // the edge we came in by
            if (state[v] == 0) {
                state[v] = 1; par[v] = u; pedge[v] = id;
                st.push_back(v);
            } else if (state[v] == 1) {              // back edge u - v closes a cycle
                UndirectedCycle c;
                for (int x = u; x != v; x = par[x]) { c.verts.push_back(x); c.ids.push_back(pedge[x]); }
                c.verts.push_back(v);                // verts: u, par(u), ..., v
                c.ids.push_back(id);                 // the back edge v - u closes it
                return c;
            }
        }
    }
    return UndirectedCycle();
}

inline vector<int> find_cycle_directed(const vector<vector<int>>& adj) {
    int n = adj.size();
    vector<int> state(n, 0), par(n, -1);
    vector<size_t> it(n, 0);
    vector<int> st;
    for (int s = 0; s < n; s++) {
        if (state[s]) continue;
        state[s] = 1;
        st.assign(1, s);
        while (!st.empty()) {
            int u = st.back();
            if (it[u] == adj[u].size()) { state[u] = 2; st.pop_back(); continue; }
            int v = adj[u][it[u]++];
            if (state[v] == 0) {
                state[v] = 1; par[v] = u;
                st.push_back(v);
            } else if (state[v] == 1) {              // u -> v with v on the stack
                vector<int> cyc;
                for (int x = u; x != v; x = par[x]) cyc.push_back(x);
                cyc.push_back(v);
                reverse(cyc.begin(), cyc.end());     // v -> ... -> u, then u -> v
                return cyc;
            }
        }
    }
    return {};
}
