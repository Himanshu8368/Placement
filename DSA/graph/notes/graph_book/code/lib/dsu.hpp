#pragma once
// dsu.hpp -- disjoint set union (union-find) with path compression + union by size.
//   DSU d(n): n singletons {0}, ..., {n-1}.
//   find(v)      : representative of v's set;
//   unite(a, b)  : merge the two sets; returns false if they were already one set;
//   same(a, b)   : are a and b in the same set?   size(v): size of v's set;
//   sets         : current number of sets.
// Complexity: O(alpha(n)) amortised per operation (effectively constant).
// Indexing  : elements 0..n-1.
// Pitfalls  : compare find(a) == find(b), never parent[a] == parent[b];
//             find is iterative (two passes), so no recursion depth issue even before
//             compression kicks in; re-create the DSU for every test case.
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> parent, sz;
    int sets;
    explicit DSU(int n = 0) : parent(n), sz(n, 1), sets(n) { iota(parent.begin(), parent.end(), 0); }

    int find(int v) {
        int r = v;
        while (parent[r] != r) r = parent[r];
        while (parent[v] != r) { int nxt = parent[v]; parent[v] = r; v = nxt; }   // compress
        return r;
    }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (sz[a] < sz[b]) swap(a, b);       // hang the smaller tree under the larger
        parent[b] = a;
        sz[a] += sz[b];
        sets--;
        return true;
    }
    bool same(int a, int b) { return find(a) == find(b); }
    int size(int v) { return sz[find(v)]; }
};
