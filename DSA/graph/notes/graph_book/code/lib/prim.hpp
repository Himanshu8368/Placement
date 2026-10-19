#pragma once
// prim.hpp -- minimum spanning tree / forest by Prim's algorithm (lazy binary heap).
//   prim(adj): adj[u] = {(v, w)} undirected (both directions present). Returns
//       weight     = total weight of a minimum spanning forest,
//       parent[v]  = the tree neighbour through which v joined (-1 for each root),
//       components = number of trees (Prim restarts from every unreached vertex).
// Idea      : grow one tree; always add the cheapest edge leaving it (cut property).
// Complexity: O(m log m). For dense graphs (m ~ n^2) the O(n^2) array version is better.
// Indexing  : vertices 0..n-1.
// Pitfalls  : skip stale heap entries (vertex already in the tree); a vertex enters the
//             tree when POPPED, not when pushed -- the reverse of BFS.
#include <bits/stdc++.h>
using namespace std;

struct PrimResult {
    long long weight = 0;
    vector<int> parent;
    int components = 0;
};

inline PrimResult prim(const vector<vector<pair<int, long long>>>& adj) {
    int n = adj.size();
    PrimResult res;
    res.parent.assign(n, -1);
    vector<char> in_tree(n, 0);
    // heap entries: (weight of the connecting edge, vertex, the tree vertex it comes from)
    priority_queue<tuple<long long, int, int>, vector<tuple<long long, int, int>>,
                   greater<tuple<long long, int, int>>> pq;
    for (int s = 0; s < n; s++) {
        if (in_tree[s]) continue;
        res.components++;
        pq.push(make_tuple(0LL, s, -1));
        while (!pq.empty()) {
            long long w; int u, from;
            tie(w, u, from) = pq.top();
            pq.pop();
            if (in_tree[u]) continue;               // stale
            in_tree[u] = 1;
            res.parent[u] = from;
            res.weight += w;
            for (const auto& e : adj[u])
                if (!in_tree[e.first]) pq.push(make_tuple(e.second, e.first, u));
        }
    }
    return res;
}
