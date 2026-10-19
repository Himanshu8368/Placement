#pragma once
// graph.hpp -- graph storage: adjacency lists (unweighted / weighted) and CSR.
//   Graph   : vector<vector<int>> adj; add_arc(u,v) directed, add_edge(u,v) undirected.
//   WGraph  : the same with (neighbour, weight) pairs, weights are long long.
//   CSR     : compressed sparse row = two flat arrays start[], to[]; neighbours of u
//             are to[start[u]] .. to[start[u+1]-1], in input order (stable build).
// Complexity: O(n + m) time and memory for all three; CSR is built in two passes.
// Indexing  : vertices 0..n-1. read_graph(..., base = 1) converts 1-indexed input.
// Pitfalls  : an undirected edge is stored twice, so the lists hold 2m entries;
//             a self-loop u-u appears twice in adj[u]; rebuild (not append) per test.
#include <bits/stdc++.h>
using namespace std;

struct Graph {
    int n;
    vector<vector<int>> adj;
    explicit Graph(int n_ = 0) : n(n_), adj(n_) {}
    void add_arc(int u, int v) { adj[u].push_back(v); }                       // u -> v
    void add_edge(int u, int v) { adj[u].push_back(v); adj[v].push_back(u); } // u -- v
};

struct WGraph {
    int n;
    vector<vector<pair<int, long long>>> adj;   // adj[u] = {(v, w), ...}
    explicit WGraph(int n_ = 0) : n(n_), adj(n_) {}
    void add_arc(int u, int v, long long w) { adj[u].push_back({v, w}); }
    void add_edge(int u, int v, long long w) { adj[u].push_back({v, w}); adj[v].push_back({u, w}); }
};

// Reads m lines "u v" and returns a Graph on n vertices (base = 1: input is 1-indexed).
inline Graph read_graph(istream& in, int n, int m, bool directed, int base = 1) {
    Graph g(n);
    for (int i = 0; i < m; i++) {
        int u, v;
        in >> u >> v;
        u -= base; v -= base;
        if (directed) g.add_arc(u, v); else g.add_edge(u, v);
    }
    return g;
}

// Reads m lines "u v w".
inline WGraph read_wgraph(istream& in, int n, int m, bool directed, int base = 1) {
    WGraph g(n);
    for (int i = 0; i < m; i++) {
        int u, v;
        long long w;
        in >> u >> v >> w;
        u -= base; v -= base;
        if (directed) g.add_arc(u, v, w); else g.add_edge(u, v, w);
    }
    return g;
}

struct CSR {
    int n;
    vector<int> start, to;      // start has n+1 entries; to has m (directed) or 2m entries
    CSR(int n_, const vector<pair<int, int>>& edges, bool directed) : n(n_), start(n_ + 1, 0) {
        for (const auto& e : edges) {               // pass 1: count out-degrees
            start[e.first + 1]++;
            if (!directed) start[e.second + 1]++;
        }
        for (int u = 0; u < n; u++) start[u + 1] += start[u];   // prefix sums
        to.assign(start[n], 0);
        vector<int> pos(start.begin(), start.end() - 1);         // next free slot of each u
        for (const auto& e : edges) {               // pass 2: place, in input order
            to[pos[e.first]++] = e.second;
            if (!directed) to[pos[e.second]++] = e.first;
        }
    }
    int deg(int u) const { return start[u + 1] - start[u]; }
    // usage: for (int i = g.start[u]; i < g.start[u + 1]; i++) { int v = g.to[i]; ... }
};
