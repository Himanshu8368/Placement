#pragma once
// weighted_dsu.hpp -- DSU with potentials ("weighted DSU"): every element v has an unknown
// value pot(v), and we learn relations  pot(b) - pot(a) = w.
//   unite(a, b, w): record pot(b) - pot(a) = w; returns false if this CONTRADICTS what
//                   is already known (the structure is then left unchanged);
//   same(a, b)    : is pot(b) - pot(a) determined?
//   diff(a, b)    : pot(b) - pot(a), only valid when same(a, b).
// Stored: d[v] = pot(v) - pot(parent[v]); find() compresses paths and sums d on the way.
// Complexity: O(log n) worst case per operation (union by size bounds the depth),
//             O(alpha(n)) amortised.
// Indexing  : elements 0..n-1. Values are long long.
// Pitfalls  : the direction matters (w is b relative to a); find() is recursive but its
//             depth is <= log2(n) thanks to union by size; works for any abelian group
//             (replace +/- by XOR for parity / XOR constraints).
#include <bits/stdc++.h>
using namespace std;

struct WeightedDSU {
    vector<int> parent, sz;
    vector<long long> d;
    explicit WeightedDSU(int n = 0) : parent(n), sz(n, 1), d(n, 0) { iota(parent.begin(), parent.end(), 0); }

    int find(int v) {                 // afterwards d[v] = pot(v) - pot(root)
        if (parent[v] == v) return v;
        int r = find(parent[v]);
        d[v] += d[parent[v]];
        parent[v] = r;
        return r;
    }
    bool same(int a, int b) { return find(a) == find(b); }
    long long diff(int a, int b) { find(a); find(b); return d[b] - d[a]; }

    bool unite(int a, int b, long long w) {
        int ra = find(a), rb = find(b);
        if (ra == rb) return d[b] - d[a] == w;
        // pot(rb) - pot(ra) = (pot(b) - d[b]) - (pot(a) - d[a]) = w - d[b] + d[a]
        long long k = w - d[b] + d[a];
        if (sz[ra] < sz[rb]) { swap(ra, rb); k = -k; }
        parent[rb] = ra;              // rb hangs under ra with d[rb] = pot(rb) - pot(ra)
        d[rb] = k;
        sz[ra] += sz[rb];
        return true;
    }
};
