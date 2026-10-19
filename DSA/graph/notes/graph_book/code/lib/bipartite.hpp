#pragma once
// bipartite.hpp -- 2-colouring of an undirected graph, with a certificate either way.
//   run(adj) returns true iff the graph is bipartite.
//     true : color[v] in {0, 1} and every edge joins different colours
//            (each component's smallest vertex gets colour 0);
//     false: odd_cycle = v0, v1, ..., v(k-1) with k odd, consecutive vertices adjacent
//            and v(k-1) adjacent to v0 (a self-loop gives k = 1).
// Idea      : BFS colours by parity of depth; an edge between two vertices of the same
//             depth parity closes the odd cycle  u -> ... -> lca <- ... <- v -> u.
// Complexity: O(n + m).
// Indexing  : vertices 0..n-1.
// Pitfalls  : test EVERY component (start a BFS from every uncoloured vertex);
//             a self-loop alone makes the graph non-bipartite.
#include <bits/stdc++.h>
using namespace std;

struct Bipartite {
    vector<int> color, odd_cycle;

    bool run(const vector<vector<int>>& adj) {
        int n = adj.size();
        color.assign(n, -1);
        odd_cycle.clear();
        vector<int> par(n, -1), depth(n, 0), q;
        for (int s = 0; s < n; s++) {
            if (color[s] != -1) continue;
            color[s] = 0;
            q.assign(1, s);
            for (size_t head = 0; head < q.size(); head++) {
                int u = q[head];
                for (int v : adj[u]) {
                    if (color[v] == -1) {
                        color[v] = color[u] ^ 1;
                        par[v] = u;
                        depth[v] = depth[u] + 1;
                        q.push_back(v);
                    } else if (color[v] == color[u]) {   // conflict: build the odd cycle
                        vector<int> left, right;          // u .. lca  and  v .. lca
                        int a = u, b = v;
                        while (depth[a] > depth[b]) { left.push_back(a); a = par[a]; }
                        while (depth[b] > depth[a]) { right.push_back(b); b = par[b]; }
                        while (a != b) {
                            left.push_back(a); a = par[a];
                            right.push_back(b); b = par[b];
                        }
                        left.push_back(a);                // the lca
                        odd_cycle = left;                 // u, ..., lca
                        for (int i = (int)right.size() - 1; i >= 0; i--)
                            odd_cycle.push_back(right[i]); // ..., v   (then v -- u closes it)
                        return false;
                    }
                }
            }
        }
        return true;
    }
};
