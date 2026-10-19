#pragma once
// gen.hpp -- seeded random generators for testing graph code (0-indexed vertices).
//   Gen g(seed);                       same seed -> same graphs (reproducible failures)
//   g.rnd(l, r)                        uniform integer in [l, r]
//   g.tree(n)                          random labelled tree, n - 1 edges
//   g.connected_graph(n, m)            random spanning tree + extra simple edges (no loops,
//                                      no multi-edges; m is capped at n(n-1)/2)
//   g.graph(n, m, loops, multi)        m random undirected edges; self-loops / parallel
//                                      edges only if allowed (m capped for simple graphs)
//   g.digraph(n, m, loops, multi)      the same for directed arcs (u, v) = u -> v
//   g.dag(n, m)                        random DAG: arcs go forward in a hidden random order
//   g.functional(n)                    f[v] uniform in 0..n-1
//   g.grid(R, C, wall_pct)             '.' / '#' grid, each cell a wall with prob wall_pct%
//   g.weighted(edges, lo, hi)          attach weights in [lo, hi] -> vector<Edge>
//   to_adj(n, edges, directed), to_wadj(n, wedges, directed)   build adjacency lists
//   print_graph / print_wgraph         contest format "n m" + edges, with a base (0 or 1)
// Edges of undirected graphs come in random order with random orientation, and vertex
// labels are shuffled, so a bug that depends on order or labels is not hidden.
#include <bits/stdc++.h>
using namespace std;

#ifndef GRAPHLIB_EDGE
#define GRAPHLIB_EDGE
struct Edge { int u, v; long long w; };
#endif

struct Gen {
    mt19937_64 rng;
    explicit Gen(unsigned long long seed) : rng(seed) {}

    long long rnd(long long l, long long r) {               // uniform in [l, r]
        return l + (long long)(rng() % (unsigned long long)(r - l + 1));
    }
    vector<int> perm(int n) {
        vector<int> p(n);
        iota(p.begin(), p.end(), 0);
        for (int i = n - 1; i > 0; i--) swap(p[i], p[rnd(0, i)]);
        return p;
    }
    void shuffle_edges(vector<pair<int, int>>& e, bool flip) {
        for (int i = (int)e.size() - 1; i > 0; i--) swap(e[i], e[rnd(0, i)]);
        if (flip)
            for (auto& x : e)
                if (rnd(0, 1)) swap(x.first, x.second);
    }

    vector<pair<int, int>> tree(int n) {
        vector<int> p = perm(n);
        vector<pair<int, int>> e;
        for (int i = 1; i < n; i++) e.push_back({p[i], p[rnd(0, i - 1)]});   // random parent
        shuffle_edges(e, true);
        return e;
    }

    vector<pair<int, int>> connected_graph(int n, int m) {
        long long cap = 1LL * n * (n - 1) / 2;
        if (m > cap) m = cap;
        if (m < n - 1) m = n - 1;
        vector<pair<int, int>> e = tree(n);
        set<pair<int, int>> have;
        for (auto& x : e) have.insert({min(x.first, x.second), max(x.first, x.second)});
        while ((int)e.size() < m) {
            int u = rnd(0, n - 1), v = rnd(0, n - 1);
            if (u == v || have.count({min(u, v), max(u, v)})) continue;
            have.insert({min(u, v), max(u, v)});
            e.push_back({u, v});
        }
        shuffle_edges(e, true);
        return e;
    }

    vector<pair<int, int>> graph(int n, int m, bool loops, bool multi, bool directed = false) {
        long long cap = directed ? 1LL * n * (n - 1) + (loops ? n : 0)
                                 : 1LL * n * (n - 1) / 2 + (loops ? n : 0);
        if (!multi && m > cap) m = cap;
        if (!loops && n < 2) m = 0;             // no edge is possible at all
        vector<pair<int, int>> e;
        set<pair<int, int>> have;
        while ((int)e.size() < m) {
            int u = rnd(0, n - 1), v = rnd(0, n - 1);
            if (u == v && !loops) continue;
            pair<int, int> key = directed ? make_pair(u, v) : make_pair(min(u, v), max(u, v));
            if (!multi && have.count(key)) continue;
            have.insert(key);
            e.push_back({u, v});
        }
        return e;
    }
    vector<pair<int, int>> digraph(int n, int m, bool loops, bool multi) { return graph(n, m, loops, multi, true); }

    vector<pair<int, int>> dag(int n, int m) {
        long long cap = 1LL * n * (n - 1) / 2;
        if (m > cap) m = cap;
        vector<int> p = perm(n);                 // p[i] = vertex at topological position i
        set<pair<int, int>> have;
        vector<pair<int, int>> e;
        while ((int)e.size() < m) {
            int a = rnd(0, n - 1), b = rnd(0, n - 1);
            if (a == b) continue;
            if (a > b) swap(a, b);
            if (have.count({a, b})) continue;
            have.insert({a, b});
            e.push_back({p[a], p[b]});           // forward in the hidden order
        }
        return e;
    }

    vector<int> functional(int n) {
        vector<int> f(n);
        for (auto& x : f) x = rnd(0, n - 1);
        return f;
    }

    vector<string> grid(int R, int C, int wall_pct) {
        vector<string> g(R, string(C, '.'));
        for (auto& row : g)
            for (auto& ch : row)
                if (rnd(1, 100) <= wall_pct) ch = '#';
        return g;
    }

    vector<Edge> weighted(const vector<pair<int, int>>& e, long long lo, long long hi) {
        vector<Edge> res;
        for (auto& x : e) res.push_back({x.first, x.second, rnd(lo, hi)});
        return res;
    }
};

inline vector<vector<int>> to_adj(int n, const vector<pair<int, int>>& e, bool directed) {
    vector<vector<int>> adj(n);
    for (auto& x : e) {
        adj[x.first].push_back(x.second);
        if (!directed) adj[x.second].push_back(x.first);
    }
    return adj;
}

inline vector<vector<pair<int, long long>>> to_wadj(int n, const vector<Edge>& e, bool directed) {
    vector<vector<pair<int, long long>>> adj(n);
    for (auto& x : e) {
        adj[x.u].push_back({x.v, x.w});
        if (!directed) adj[x.v].push_back({x.u, x.w});
    }
    return adj;
}

inline void print_graph(ostream& out, int n, const vector<pair<int, int>>& e, int base = 1) {
    out << n << " " << e.size() << "\n";
    for (auto& x : e) out << x.first + base << " " << x.second + base << "\n";
}

inline void print_wgraph(ostream& out, int n, const vector<Edge>& e, int base = 1) {
    out << n << " " << e.size() << "\n";
    for (auto& x : e) out << x.u + base << " " << x.v + base << " " << x.w << "\n";
}

// a short human-readable dump, used by the tests when they report a failing case
inline string dump_edges(int n, const vector<pair<int, int>>& e) {
    ostringstream os;
    os << "n=" << n << " m=" << e.size() << " edges:";
    for (auto& x : e) os << " " << x.first << "-" << x.second;
    return os.str();
}
inline string dump_wedges(int n, const vector<Edge>& e) {
    ostringstream os;
    os << "n=" << n << " m=" << e.size() << " edges:";
    for (auto& x : e) os << " " << x.u << "-" << x.v << "(" << x.w << ")";
    return os.str();
}
