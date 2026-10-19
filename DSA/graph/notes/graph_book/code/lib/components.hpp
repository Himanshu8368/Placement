#pragma once
// components.hpp -- connected components of an undirected graph.
//   run(adj): count       = number of components,
//             comp[v]     = component id in 0..count-1, numbered in order of the
//                           smallest vertex of each component,
//             size[c]     = number of vertices in component c.
// Complexity: O(n + m) with an iterative BFS (no recursion-depth issue).
// Indexing  : vertices 0..n-1.
// Pitfalls  : adj must be symmetric (add both directions); on a directed graph this
//             computes WEAK components only if you add the reverse arcs yourself --
//             strongly connected components are a different algorithm (Chapter 19).
#include <bits/stdc++.h>
using namespace std;

struct Components {
    int count = 0;
    vector<int> comp, size;

    void run(const vector<vector<int>>& adj) {
        int n = adj.size();
        count = 0;
        comp.assign(n, -1);
        size.clear();
        vector<int> q;
        for (int s = 0; s < n; s++) {
            if (comp[s] != -1) continue;
            comp[s] = count;
            q.assign(1, s);
            for (size_t head = 0; head < q.size(); head++)
                for (int v : adj[q[head]])
                    if (comp[v] == -1) {
                        comp[v] = count;
                        q.push_back(v);
                    }
            size.push_back(q.size());
            count++;
        }
    }
};
