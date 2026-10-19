#pragma once
// kruskal.hpp -- minimum spanning tree / forest by Kruskal's algorithm.
//   kruskal(n, edges): edges are undirected {u, v, w}. Returns
//       weight     = total weight of a minimum spanning FOREST,
//       chosen     = indices of the chosen edges (n - components of them),
//       components = number of trees (1 iff the graph is connected).
// Idea      : sort edges by weight; take an edge iff it joins two different trees (DSU).
// Complexity: O(m log m) for the sort, O(m alpha(n)) for the DSU part.
// Indexing  : vertices 0..n-1, edge indices refer to the input vector.
// Pitfalls  : check components == 1 before calling the result a spanning TREE;
//             sum weights in long long; self-loops are never chosen.
#include <bits/stdc++.h>
using namespace std;

#ifndef GRAPHLIB_EDGE
#define GRAPHLIB_EDGE
struct Edge { int u, v; long long w; };
#endif

struct MSTResult {
    long long weight = 0;
    vector<int> chosen;
    int components = 0;
};

inline MSTResult kruskal(int n, const vector<Edge>& edges) {
    vector<int> idx(edges.size());
    iota(idx.begin(), idx.end(), 0);
    stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return edges[a].w < edges[b].w; });
    vector<int> parent(n), sz(n, 1);                // a minimal DSU, kept local
    iota(parent.begin(), parent.end(), 0);
    auto find = [&](int v) {
        while (parent[v] != v) { parent[v] = parent[parent[v]]; v = parent[v]; }   // halving
        return v;
    };
    MSTResult res;
    res.components = n;
    for (int i : idx) {
        int a = find(edges[i].u), b = find(edges[i].v);
        if (a == b) continue;                       // would close a cycle
        if (sz[a] < sz[b]) swap(a, b);
        parent[b] = a;
        sz[a] += sz[b];
        res.weight += edges[i].w;
        res.chosen.push_back(i);
        res.components--;
    }
    return res;
}
